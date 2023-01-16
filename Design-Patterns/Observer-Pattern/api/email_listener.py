from lib.send_email import send_email
from .event import subscribe
from lib.db import Trades
from states.trade import TradeActions


def handle_bought_security(trade: Trades) -> None:
    send_email(
        email="test@gmail.com",
        name="Warren Buffet",
        subject=f"Bought {trade.ticker}",
        body=f"""Bought {trade.quantity} units of {trade.ticker} at {trade.price} """,
    )


def handle_sold_security(trade: Trades) -> None:
    send_email(
        email="test@gmail.com",
        name="Warren Buffet",
        subject=f"Sold {trade.ticker}",
        body=f"""Sold {trade.quantity} units of {trade.ticker} at {trade.price} """,
    )


def setup_email_event_handlers() -> None:
    subscribe(TradeActions.BUY, handle_bought_security)
    subscribe(TradeActions.SELL, handle_sold_security)
