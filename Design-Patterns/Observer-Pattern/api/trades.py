from .event import post_event
from lib.db import _buy_security, _sell_security
from states.trade import TradeActions


def buy_security(ticker: str, quantity: int, price: float):
    # create an entry in the database
    user = _buy_security(ticker, quantity, price)
    post_event(TradeActions.BUY, user)


def sell_security(ticker: str, quantity: int, price: float):
    user = _sell_security(ticker, quantity, price)
    post_event(TradeActions.SELL, user)
