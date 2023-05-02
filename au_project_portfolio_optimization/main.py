import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import pandas_datareader as pdr


# Matplotlib Style
plt.style.use("ggplot")


def get_prices(
    stocks: list[str], step_size: int, period: tuple[str, str]
) -> tuple[list[str], DataFrame]:
    """
    The function 'get_prices' will return a list of the names of the stocks
    and get open stock prices from 'StooqDailyReader' and return it as a Panda Dataframe

    Parameters
        stocks: A list of the names of the stocks
        step_size: A positiv interger greater than 0
        period: A tuple with start date at index 0 and end date at index 1

    Returns
        stocks: A list of the names of the stocks
        df_prices: Pandas DataFrame where p[j, i] represents the opening price of stock i
            at time step j
    """
    assert step_size > 0, "Step size must be greater than 0"

    df = pd.DataFrame()
    for stock in stocks:
        df_stooq = pdr.stooq.StooqDailyReader(
            stock, start=period[0], end=period[1]
        ).read()
        df = pd.concat((df, df_stooq[["Open"]]), axis=1)
    # We rename the columns of the DataFrame
    df.columns = stocks

    df_prices = df.iloc[::step_size]

    return stocks, df_prices


def plot_stocks(df_prices: DataFrame) -> None:
    """
    Plot stock prices with matplotlib

    Parameters
        df_prices: Panda DataFrame with the open prices
    """
    df_prices.plot()
    plt.legend()
    plt.title("Stock prices")
    plt.ylabel("Price $")
    plt.xlabel("Date")
    plt.show()


if __name__ == "__main__":
    # ------------------------------
    #   Subtask 1)
    # ------------------------------
    stocks = ["MSFT", "GOOGL", "FLWS", "FOX", "UHAL"]
    step_size = 1
    period = ("2000-01-01", "2023-01-01")

    stocks, df_prices = get_prices(stocks, step_size, period)
    plot_stocks(df_prices)

    # ------------------------------
    #   Subtask 2)
    # ------------------------------

    # ------------------------------
    #   Subtask 3)
    # ------------------------------
