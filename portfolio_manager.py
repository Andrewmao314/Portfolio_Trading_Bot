import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PortfolioManager:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

    def update_portfolio(self, target_portfolio):
        """
        Update the Alpaca portfolio based on the target portfolio allocation.
        """
        logging.info("Updating Alpaca portfolio...")

        try:
            # Get current positions
            current_positions = {p.symbol: p.qty for p in self.trading_client.get_all_positions()}

            # Get account information
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)

            logging.info(f"Current buying power: ${buying_power:.2f}")

            # Calculate the total portfolio value
            total_portfolio_value = sum(float(p.market_value) for p in self.trading_client.get_all_positions())
            total_portfolio_value += buying_power

            logging.info(f"Total portfolio value: ${total_portfolio_value:.2f}")

            # Place orders to adjust the portfolio
            for symbol, target_weight in target_portfolio.items():
                target_value = total_portfolio_value * target_weight
                current_value = float(current_positions.get(symbol, 0))

                if symbol in current_positions:
                    current_price = float(self.trading_client.get_open_position(symbol).current_price)
                else:
                    # If we don't have a position, get the latest quote
                    latest_quote = self.trading_client.get_stock_latest_quote(symbol)
                    current_price = float(latest_quote.ask_price)  # Use ask price for buying

                order_value = target_value - current_value

                if abs(order_value) > 1:  # Only place orders for significant changes
                    order_qty = round(order_value / current_price, 2)  # Round to 2 decimal places for fractional shares
                    side = OrderSide.BUY if order_value > 0 else OrderSide.SELL

                    try:
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=abs(order_qty),
                            side=side,
                            time_in_force=TimeInForce.DAY
                        )
                        order = self.trading_client.submit_order(order_request)
                        logging.info(f"Placed {side.name} order for {abs(order_qty)} shares of {symbol}")
                    except Exception as e:
                        logging.error(f"Error placing order for {symbol}: {e}")

            logging.info("Portfolio update completed.")

        except Exception as e:
            logging.error(f"Error updating portfolio: {e}")
