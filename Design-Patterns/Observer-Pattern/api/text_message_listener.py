from states.trade import TradeActions
from lib.send_text_message import send_text_message
from .event import subscribe
from lib.db import Trades


def handle_bought_security(trade: Trades) -> None:
    return send_text_message(name="Warren Buffet", number="123-456-7890", trade=trade)


def handle_sold_security(trade: Trades) -> None:
    return send_text_message(name="Warren Buffet", number="123-456-7890", trade=trade)


def setup_text_message_event_handler() -> None:
    subscribe(TradeActions.BUY, handle_bought_security)
    subscribe(TradeActions.SELL, handle_sold_security)
