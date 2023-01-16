from api.email_listener import setup_email_event_handlers
from api.text_message_listener import setup_text_message_event_handler
from api.trades import buy_security, sell_security

setup_email_event_handlers()
setup_text_message_event_handler()

if __name__ == "__main__":
    buy_security("googl", 1000, 543.59)
    sell_security("AAPL", 1000, 304.80)
