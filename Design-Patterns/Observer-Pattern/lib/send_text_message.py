from lib.db import Trades


def send_text_message(name: str, number: str, trade: Trades) -> None:
    print(
        f"""\n--- Sending text to {name} at number {number} \n
Message: Purchase {trade.quantity} of {trade.ticker} for {trade.price}"""
    )
