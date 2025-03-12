# import QuantLib as ql
import aadc
import aadc.quantlib as ql
import time
import random
random.seed(42)  # Set random seed for reproducibility
print(ql.__version__)
import numpy as np
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
            swaps.append( curve.create_swap(fixed_rate=random.uniform(0.005, 0.05), period= '10Y' ) )
    return  swaps, curves


def with_aadc_kernel(func):
    _cache = {}
    def wrapper(swaps, curves, curve_rates):
        key = func.__name__
        if key not in _cache:
            kernel = aadc.Kernel()
            kernel.start_recording()
            curve_rates = []
            curve_args  = []
            for curve in curves:
                zero_rates = aadc.array(np.zeros(len(curve.tenors)))
                curve_rates.append(zero_rates)
                curve_args.append(zero_rates.mark_as_input())

            res = func(swaps, curves, curve_rates)
            res_aadc = res.mark_as_output()
            request = { res_aadc: [] }
            kernel.stop_recording()
            _cache[key] = (kernel, request, curve_args)

        # Subsequent calls: use cached kernel
        kernel, request, curve_args = _cache[key]
        inputs = {}
        for curve_arg, rates in zip(curve_args, curve_rates):
            for arg_point, rate in zip( curve_arg, rates ):
                inputs[arg_point] = rate

        r = aadc.evaluate(kernel, request, inputs, aadc.ThreadPool(1))
        res_aadc = list(request.keys())[0]
        result  = r[0][res_aadc]
        if len(result) == 1:
            result = float(result[0])
        return result
    
    return wrapper

@with_aadc_kernel
def price_portfolio(swaps, curves, curve_rates):
    npv = 0.
    for curve, rates in zip(curves, curve_rates):
        for quote, rate in zip(curve.quotes, rates):
            quote.setValue(rate)
    for swap in swaps:
        npv += swap.NPV()
    return npv

def do_it():
    trade_count = 3333
    params = [(SofrCurve, trade_count), (EstrCurve, trade_count), (SoniaCurve, trade_count)]
    swaps, curves = build_portfolio(params)
    curve_rates = []
    for curve in curves:
        rates = []
        for _ in curve.tenors:
            rates.append( 0.0025 + 0.005 * 0.02 * random.randint(0, 99))
        curve_rates.append(rates)

    res = price_portfolio(swaps, curves, curve_rates)

    curve_rates = []
    for curve in curves:
        rates = []
        for _ in curve.tenors:
            # rates.append( 0.0025 + 0.005 * 0.02 * random.randint(0, 99))
            rates.append(np.random.randint(0, 99, 10) *0.005 * 0.02 +  0.0025 )

        curve_rates.append(rates)
        
    start  = time.time()
    res = price_portfolio(swaps, curves, curve_rates)
    pprint(res)
    print("Price time: ", time.time() - start)


if __name__ == "__main__":
    do_it()