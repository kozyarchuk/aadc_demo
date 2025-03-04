# import QuantLib as ql
import aadc
import aadc.quantlib as ql
import pickle
import time
import sys
from memusage import get_memory_usage
import zipfile
import random
print(ql.__version__)
import numpy as np
from collections import defaultdict
from pprint import pprint

class Curve:
    def __init__(self, rates, tenors):
        self.tenors = tenors
        self.rates = rates
        self.quotes = [ql.SimpleQuote(rate) for rate in self.rates]    
        self.yts = ql.RelinkableYieldTermStructureHandle()
        self.index = self.get_index()
        helpers = []
        for tenor,quote in zip(self.tenors, self.quotes):
            helpers.append(ql.OISRateHelper(2, ql.Period(tenor), 
                                            ql.QuoteHandle(quote), self.index))
        self.curve = ql.PiecewiseLinearZero(0, self.CALENDAR, helpers, ql.Actual360())
        self.curve.enableExtrapolation()
        self.yts.linkTo(self.curve)
        self.engine = ql.DiscountingSwapEngine(self.yts)

    def get_index(self):
        raise NotImplementedError

    def create_swap(self, fixed_rate=0.02, period='10Y'):
        swap = ql.MakeOIS(ql.Period(period), self.index, fixed_rate, nominal=1_000_000)
        swap.setPricingEngine(self.engine)
        return swap

class SofrCurve(Curve):
    CALENDAR = ql.UnitedStates(ql.UnitedStates.FederalReserve)
    def get_index(self):
        return ql.Sofr(self.yts)

class EstrCurve(Curve):
    CALENDAR = ql.TARGET()
    def get_index(self):
        return ql.Estr(self.yts)

class SoniaCurve(Curve):
    CALENDAR = ql.UnitedKingdom()

    def get_index(self):
        return ql.Sonia(self.yts)

def build_portfolio(params):
    tenors = ['1D', '1M', '3M', '6M', '1Y', '3Y', '5Y', '10Y', '20Y', '30Y']
    rates = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.025, 0.03, 0.035]
    swaps = []
    curves = []
    for klass, count in params:
        curve = klass(rates, tenors)
        curves.append(curve)
        for _ in range(count):
            swaps.append( (curve, curve.create_swap(fixed_rate=random.uniform(0.005, 0.05), period= random.choice(tenors) ) ) )
    return  swaps, curves

def price_portfolio(swaps, curves, curve_rates, results):
    for curve, rates in zip(curves, curve_rates):
        for quote, rate in zip(curve.quotes, rates):
            quote.setValue(rate)
    for curve, swap in swaps:
        npv = swap.NPV()
        results[curve.index.currency().code()] += npv

def record_kernel(swaps, curves):
    kernel = aadc.Kernel()
    kernel.start_recording()

    curve_rates =[]
    curve_args = []
    for curve in curves:
        zero_rates = aadc.array(np.zeros(len(curve.tenors)))
        curve_rates.append(zero_rates)
        curve_args.append(zero_rates.mark_as_input())

    results = defaultdict(lambda: aadc.idouble(0.0))
    price_portfolio(swaps, curves, curve_rates, results)
    reqest = {}
    result_template = {}
    for ccy, ccy_npv in results.items():
        ccy_npv_res = ccy_npv.mark_as_output()
        result_template[ ccy ] = ccy_npv_res
        reqest[ ccy_npv_res ] = []
    kernel.stop_recording()
    kernel.print_passive_extract_locations()

    return (kernel, reqest, result_template, curve_args)

def price_portfolio_aadc(kernel, request, curve_args, result_template):
    inputs = {}
    for curve_arg in curve_args:
        for arg_point in curve_arg:
            inputs[arg_point] = 0.0025 + 0.005 * 0.02 * random.randint(0, 99)
    start = time.time()

    r = aadc.evaluate(kernel, request, inputs, aadc.ThreadPool(1))
    for ccy, ccy_npv in list(result_template.items()):
        result_template[ccy] = r[0][ccy_npv][0]
    print("Calculation time: ", time.time() - start)

def do_it():
    trade_count = 10
    params = [(SofrCurve, trade_count), (EstrCurve, trade_count), (SoniaCurve, trade_count)]
    swaps, curves = build_portfolio(params)

    start  = time.time()
    kernel, request, result_template, curve_args = record_kernel(swaps, curves)
    print("Recording time: ", time.time() - start)

    price_portfolio_aadc( kernel, request, curve_args, result_template)
    pprint(result_template)

do_it()