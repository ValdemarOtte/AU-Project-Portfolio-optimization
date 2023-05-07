import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas.core.frame import DataFrame
import pandas_datareader as pdr
from scipy.stats import norm


# Matplotlib Style
plt.style.use("ggplot")


# ------------------------------
#   Subtask 1)
# ------------------------------
def get_prices(
    stocks: list[str], step_size: int, period: tuple[str, str]
) -> tuple[list[str], DataFrame]:
    """
    The function 'get_prices' will get open stock prices from 'StooqDailyReader'
    and return it as a Panda Dataframe and the names of the stocks

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

    df = df.iloc[::-1]

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


# ------------------------------
#   Subtask 2)
# ------------------------------
def calculate_r(df_prices: DataFrame) -> ndarray:
    """
    Parameters
        df_prices: Panda DataFrame with the open prices

    Return
        r: Array with the fractional reward of stock i at time step j.
    """
    # We use the function pct_change from Pandas. This give us the datafram where
    # df_{j, i} = (p_{j, i} - p_{j-1, i}) / p_{j, i}. This is not the right represents
    # For returning the righ format, we skift the df ones up and delete the last row
    df = df_prices.pct_change()
    df = df.shift(periods=-1)
    df.drop(df.tail(1).index, inplace=True)

    # We want to change to format from Panda DataFrame to numpy,
    # because the calculate later on will be easy to compute
    r = np.array(df)
    return r


def calculate_mu(r: ndarray) -> ndarray:
    """
    Parameters
        r: Array with the fractional reward of stock i at time step j.

    Return
        mu: Array with the estimated means for the stocks
    """
    mu = (1 / r.shape[0]) * np.sum(r, axis=0)
    return mu


def calculate_sigma(r: ndarray, mu: ndarray) -> ndarray:
    """
    Parameters
        r: Array with the fractional reward of stock i at time step j.
        mu: Array with the estimated means for the stocks

    Returns
        sigma: Covariance matrix between the stocks
    """
    sigma = (1 / r.shape[0]) * ((np.matmul((r - mu).transpose(), ((r - mu)))))
    return sigma


def plot_pdf_norm(stocks: list[str], mu: ndarray, sigma: ndarray) -> None:
    """
    Plots the pdf of the rate of return of each stock

    Parameters:
        stocks: A list of the names of the stocks
        mu: Array with the estimated means for the stocks
        sigma: Covariance matrix between the stocks
    """
    # We use numpy's linspace to get a list of points, which our norm.pdf needs
    # No matter what, the the plot will show 95.3% of the distribution
    x = np.linspace(
        min(mu) - 2 * max(np.diag(sigma)), max(mu) + 2 * max(np.diag(sigma))
    )
    for i, stock in enumerate(stocks):
        plt.plot(x, norm.pdf(x, mu[i], sigma[i][i]))

    plt.legend(stocks)
    plt.title("Pdf of rate of return on stocks")
    plt.xlabel("Rate of returns")
    plt.show()


# ------------------------------
#   Subtask 3)
# ------------------------------


if __name__ == "__main__":
    # ------------------------------
    #   Subtask 1)
    # ------------------------------
    stocks = ["MSFT", "GOOGL", "FLWS", "FOX", "UHAL"]
    step_size = 7
    period = ("2012-01-01", "2021-01-01")

    stocks, df_prices = get_prices(stocks, step_size, period)
    plot_stocks(df_prices)

    # ------------------------------
    #   Subtask 2)
    # ------------------------------
    r = calculate_r(df_prices)
    mu = calculate_mu(r)
    sigma = calculate_sigma(r, mu)
    plot_pdf_norm(stocks, mu, sigma)

    # ------------------------------
    #   Subtask 3)
    # ------------------------------
