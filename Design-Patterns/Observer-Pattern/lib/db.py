from states.trade import TradeActions


# the databse of trades
all_trades = []


class Trades:
    def __init__(
        self, action: TradeActions, ticker: str, quantity: int, price: float
    ) -> None:
        self.action = action
        self.ticker = ticker
        self.quantity = quantity
        self.price = price

    def __repr__(self):
        return (
            f"{self.action} {self.quantity} units of { self.ticker } for {self.price}"
        )


def _buy_security(ticker: str, quantity: int, price: float):
    trade = Trades(TradeActions.BUY, ticker, quantity, price)
    all_trades.append(trade)
    return trade


def _sell_security(ticker: str, quantity: int, price: float):
    trade = Trades(TradeActions.SELL, ticker, quantity, price)
    all_trades.append(trade)
    return trade
