import statistics


def get_indicators(close_prices, offset):
    sma_list = []
    u_bollinger_band = []
    l_bollinger_band = []
    for i in range(offset, len(close_prices)):
        price_range = close_prices[i - offset:i]
        sma = statistics.mean(price_range)
        standard_deviation = statistics.stdev(price_range)
        sma_list.append(sma)
        u_bollinger_band.append(sma + 2 * standard_deviation)
        l_bollinger_band.append(sma - 2 * standard_deviation)
    return sma_list, u_bollinger_band, l_bollinger_band
