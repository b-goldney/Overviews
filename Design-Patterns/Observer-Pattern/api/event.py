from states.trade import TradeActions

subscribers = dict()


def subscribe(event_type: TradeActions, fn):
    if not event_type in subscribers:
        subscribers[event_type] = []
    subscribers[event_type].append(fn)


def post_event(event_type: TradeActions, data):
    if not event_type in subscribers:
        return
    print("----------- BEGINNING NOTIFICATIONS -----------")
    for fn in subscribers[event_type]:
        fn(data)
    print("----------- END NOTIFICATIONS ----------- \n \n")
