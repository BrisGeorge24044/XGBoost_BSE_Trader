import xgboost as xgb
import numpy as np
from collections import deque
import pandas as pd

class TraderPT2(Trader):
    def __init__(self, ttype, tid, balance, parameters, time):
        super().__init__(ttype, tid, balance, parameters, time) # Initialises parent Trader class

        # Loads the trained model
        self.model = xgb.Booster()
        self.model.load_model("xgb_trader_tuned_new_syn.ubj")

        # Stores price and volume history for features
        self.price_history = deque(maxlen=15)  # For lag features, EMA and RSI
        self.volume_history = deque(maxlen=3)  # For volume_change

        self.last_order = None
        self.birthbalance = balance
        self.last_buy_time = None # time when last unit was bought for holding logic

        self.params = parameters or {}

        self.job = 'Buy' # Trader initially set to buy
        self.last_purchase_price = 0 # last buy price

    def bookkeep(self, time, trade, order, vrbs):

        # Fallback if the active order is not in self.orders
        fallback_order = self.last_order if self.last_order else None
        active_order = self.orders[0] if self.orders else fallback_order
        if not active_order:
            return

        price = trade['price']

        # when a buy is executed, balance is updated and job switches to 'sell'
        if active_order.otype == 'Bid':
            self.balance -= price
            self.last_purchase_price = price
            self.job = 'Sell'
            self.last_buy_time = time

        # when a sell is executed, balance is updated and job switches to 'buy'
        elif active_order.otype == 'Ask':
            self.balance += price
            self.last_purchase_price = 0
            self.job = 'Buy'

        # Updates profit statistics
        self.profit = self.balance - self.birthbalance
        self.profitpertime = self.balance / (time - self.birthtime) if (time - self.birthtime) > 0 else 0

        # Logs the trade in the blotter and removes the order
        self.blotter.append(trade)
        self.del_order(order)

    # Calculates RSI in a window of 14 time steps
    def _compute_rsi(self, prices, window=14):
        if len(prices) < window + 1:
            return 50  # neutral
        delta = np.diff(prices)
        gain = np.clip(delta, 0, None)
        loss = -np.clip(delta, None, 0)
        avg_gain = np.mean(gain[-window:])
        avg_loss = np.mean(loss[-window:])
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # Computes the model features
    def _extract_features(self):
        # returns matrix of zeros if there is not enough history
        if len(self.price_history) < 15 or len(self.volume_history) < 3:
            return xgb.DMatrix(np.zeros((1, 11)))

        # Sets up arrays for price and volume data
        price_array = np.array(self.price_history)
        volume_array = np.array(self.volume_history)

        # Computes Features
        price_t = price_array[-1]
        price_t_1 = price_array[-2]
        price_t_2 = price_array[-3]
        price_t_3 = price_array[-4]
        ema_5 = pd.Series(price_array).ewm(span=5, adjust=False).mean().iloc[-1]
        momentum = price_t - price_array[-4]
        volatility_5 = np.std(price_array[-5:])
        volume_change = volume_array[-1] - volume_array[-2]
        price_diff = price_array[-1] - price_array[-2]
        price_diff2 = price_array[-1] - price_array[-3]
        rsi_14 = self._compute_rsi(price_array, window=14)

        # creates DataFrame for features
        df = pd.DataFrame([[
            price_t, price_t_1, price_t_2, price_t_3, ema_5,
            momentum, volatility_5, volume_change,
            price_diff, price_diff2, rsi_14
        ]], columns=[
            'price_t', 'price_t-1', 'price_t-2', 'price_t-3', 'ema_5',
            'momentum', 'volatility_5', 'volume_change',
            'price_diff', 'price_diff2', 'rsi_14'
        ])

        # returns feature matrix
        return xgb.DMatrix(df)

    # Returns the last order
    def getorder(self, time, countdown, lob):
        if self.orders:
            return self.orders[-1]
        return None

    # Contains buy and sell logic
    def respond(self, time, lob, trade, vrbs):
        # Trader waits for 5 minutes before trading
        if time < 300 or len(self.orders) > 0:
            return

        # Calculates mid-price between best bid and best ask
        best_bid = lob['bids']['best'] if lob['bids']['best'] else 0
        best_ask = lob['asks']['best'] if lob['asks']['best'] else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        self.price_history.append(mid_price)

        # Estimates market volume from the LOB
        total_volume = 0
        if 'Q' in lob.get('bids', {}):
            total_volume += sum(lob['bids']['Q'])
        if 'Q' in lob.get('asks', {}):
            total_volume += sum(lob['asks']['Q'])
        self.volume_history.append(total_volume)

        # Ensures sufficient data has been collected
        if len(self.price_history) < 15 or len(self.volume_history) < 3:
            return

        # Predicts next price using the model and features
        features = self._extract_features()
        predicted_price = self.model.predict(features)[0]

        # Clamps predictions to + or - 5% of the current mid-price to avoid extreme predictions
        if mid_price > 0:
            predicted_price = max(predicted_price, mid_price * 0.95)
            predicted_price = min(predicted_price, mid_price * 1.05)

        if vrbs:
            print(
                f"[{self.tid}] Responding at t={time:.2f} | Predicted={predicted_price:.2f} | Mid={mid_price:.2f}")

        # Buy logic
        if self.job == 'Buy':
            # Buys if predicted price is greater than (10 + best ask)
            if best_ask > 0 and predicted_price > (10 + best_ask):
                # Places a bid of best_ask + 1 to cross the spread
                bid_price = best_ask + 1
                # Allows bid if trader balance is greater than the bid price
                if bid_price <= self.balance:
                    order = Order(self.tid, 'Bid', int(bid_price), 1, time, 'PT')
                    self.orders = [order]

        # Sell logic
        elif self.job == 'Sell':
            # Tracks how long the unit has been held
            hold_duration = time - self.last_buy_time if self.last_buy_time is not None else 0
            # Sets sell price at purchase price + 5
            ask_price = self.last_purchase_price + 5

            # Sells if acceptable profit margin is hit
            if best_bid > 0 and predicted_price >= ask_price and best_bid >= ask_price:
                order = Order(self.tid, 'Ask', int(ask_price), 1, time, 'PT')
                self.orders = [order]
                if vrbs:
                    print(f"[{self.tid}] Submitting Ask at {ask_price}")

            # After holding the unit for 2000 time steps, the trader is set to sell at a break even price
            elif hold_duration >= 2000 and best_bid >= self.last_purchase_price:
                order = Order(self.tid, 'Ask', int(best_bid), 1, time, 'PT')
                self.orders = [order]
                if vrbs:
                    print(f"[{self.tid}] Break-even sale at {best_bid} after 2000s")

            # After holding the unit for 4000 time steps, the trader is set to sell at the best current bid price
            elif hold_duration >= 4000 and best_bid > 0:
                order = Order(self.tid, 'Ask', int(best_bid), 1, time, 'PT')
                self.orders = [order]
                if vrbs:
                    print(f"[{self.tid}] Stop-loss sale at {best_bid} after 4000s")