from src.StockPrediction.nnconfig import NNConfig
from src.StockPrediction.prediction_data import PredictionData


class StrategyConfig:
    value = 10000  # Starting value of account
    trade_threshold = 0.05  # Percentage profit/loss required for a trade
    cost_per_deal = 6  # Cost of making one trade
    # prediction_weights = [0.1, 0.2, 0.7]  # Weight to apply to previous predictions when deciding to trade or not
    prediction_weights = [0, 0, 1.0]  # Weight to apply to previous predictions when deciding to trade or not


class StrategyCalculator:
    def __init__(self, config: NNConfig, prediction_data: PredictionData):
        self.nn_config = config
        self.prediction_data = prediction_data
        self.current_signal = "Hold"
        self.strat_config = StrategyConfig()

    def get_buy_and_hold_return(self, targets):
        first_open = targets[0]
        last_open = targets[-1]
        profit = last_open / first_open
        return profit * self.strat_config.value

    def get_model_return(self, predictions):
        current_sma = self.prediction_data.get_prices(True, True, len(predictions))
        current_prices = self.prediction_data.get_prices(True, False, len(predictions))
        signals = []
        for index, prediction in enumerate(predictions):
            historical_values = self._get_history(signals)
            signal, profit = self._calculate_trading_signal(prediction, current_sma[index], historical_values)
            signals.append((signal, profit))
        return self._calculate_trades(signals, current_prices), signals

    def _calculate_trades(self, signals, prices):
        is_holding = False
        cash_value = self.strat_config.value
        share_value = 0
        shares_owned = 0
        for index, signal in enumerate(signals):
            share_value = prices[index] * shares_owned
            if signal[0] == "Buy" and not is_holding:
                is_holding = True
                cash_value -= self.strat_config.cost_per_deal
                shares_owned = cash_value / prices[index]
                share_value = cash_value
                cash_value = 0
            if signal[0] == "Sell" and is_holding:
                is_holding = False
                share_value -= self.strat_config.cost_per_deal
                cash_value = share_value
                share_value = 0
                shares_owned = 0

        return cash_value + share_value

    def _get_history(self, signals):
        if len(signals) == 0:
            return [("Wait", 0), ("Wait", 0)]
        elif len(signals) == 1:
            return [("Wait", 0), signals[0]]
        else:
            return [signals[-2], signals[-1]]

    def _calculate_trading_signal(self, prediction, current_sma, history):
        price_diff = prediction - current_sma
        trade_cost = self.strat_config.cost_per_deal / self.strat_config.value
        raw_profit = price_diff / current_sma
        adjusted_profit = raw_profit - trade_cost
        profit_with_hist = history[0][1] * self.strat_config.prediction_weights[0] + history[1][1] * \
                           self.strat_config.prediction_weights[1] + adjusted_profit * \
                           self.strat_config.prediction_weights[2]

        if (profit_with_hist * 100) >= self.strat_config.trade_threshold:
            signal = "Buy"
        # Shouldn't sell just because the cost of selling pushes it over the threshold
        elif ((profit_with_hist + trade_cost) * 100) <= -self.strat_config.trade_threshold:
            signal = "Sell"
        else:
            signal = history[1][0]

        if signal == "Hold" or signal == "Wait":
            return signal, price_diff
        elif signal == history[1][0]:
            if signal == "Buy":
                return "Hold", price_diff
            else:
                return "Wait", price_diff
        elif signal == "Buy" and history[1][0] == "Hold":
            return "Hold", price_diff
        elif signal == "Sell" and history[1][0] == "Wait":
            return "Wait", price_diff
        else:
            return signal, adjusted_profit
