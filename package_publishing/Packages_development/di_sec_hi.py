def second_higest(data: list) -> int:
    order = sorted(set(data), reverse=True)
    return order[1]