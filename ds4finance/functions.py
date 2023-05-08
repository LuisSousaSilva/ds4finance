import numpy as np
import pandas as pd
from datetime import date
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

def compute_std_dev(dataframe, freq="auto"):
    """
    Compute the annualized standard deviation of a given DataFrame of prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    freq : str, optional
        Frequency of the data.
        Possible values: 'daily', 'monthly', 'quarterly', 'auto'.
        Default is 'auto', which calculates the scaling factor based on the average
        frequency in the data.

    Returns
    -------
    pandas.Series
        A pandas Series containing the annualized standard deviation
        for each column in the input DataFrame.
    """

    # Define a dictionary to map frequency to the appropriate scaling factor
    scaling_factors = {
        "daily": np.sqrt(252),
        "monthly": np.sqrt(12),
        "quarterly": np.sqrt(4),
    }

    # If freq is 'auto', calculate the scaling factor based on the average frequency in
    #  the data
    if freq == "auto":
        avg_frequency = dataframe.index.to_series().diff().mean()
        sessions_per_year = pd.Timedelta(365, unit="D") / avg_frequency
        scaling_factor = np.sqrt(sessions_per_year)
    elif freq in scaling_factors:
        scaling_factor = scaling_factors[freq]
    else:
        raise ValueError(
            "Invalid frequency. Allowed values: 'daily', 'monthly', 'quarterly', 'auto'"
        )

    # Compute the percentage change in the price data
    pct_change = dataframe.pct_change()

    # Calculate the standard deviation, scale it according to the scaling factor,
    # and convert to percentage
    annualized_std_dev = pct_change.std().mul(scaling_factor).mul(100)

    return annualized_std_dev


def compute_growth_index(dataframe, initial_value=100, initial_cost=0, ending_cost=0):
    """
    Compute the growth index of a given DataFrame of prices, accounting for initial and
    ending costs.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    initial_value : float, optional
        Initial value of the investment. Default is 100.
    initial_cost : float, optional
        Initial cost as a percentage of the investment. Default is 0.
    ending_cost : float, optional
        Ending cost as a percentage of the investment. Default is 0.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the growth index for each column in the input DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    initial_cost = initial_cost / 100
    ending_cost = ending_cost / 100

    GR = ((1 + dataframe.pct_change()).cumprod()) * (initial_value * (1 - initial_cost))
    GR.iloc[0] = initial_value * (1 - initial_cost)
    GR.iloc[-1] = GR.iloc[-1] * (1 * (1 - ending_cost))
    return GR


def compute_drawdowns(dataframe):
    """
    Compute the drawdowns of a given DataFrame of prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the drawdowns for each column in the input DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Compute the drawdowns as a percentage
    drawdowns = (dataframe / dataframe.cummax() - 1) * 100

    return drawdowns


def compute_return(dataframe: pd.DataFrame, years: int = None) -> pd.Series:
    """
    Compute the return of a time series given a DataFrame of prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. The index should be of datetime type.
    years : int, optional
        Number of years to filter the data before computing the return. If not provided,
        the return will be calculated using the entire DataFrame.

    Returns
    -------
    pandas.Series
        A pandas Series containing the return for each column in the input DataFrame.
    """
    # Validate input
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input 'dataframe' must be a pandas DataFrame.")

    # Filter data by the number of years if specified
    if years is not None:
        if not isinstance(years, int) or years < 1:
            raise ValueError("Input 'years' must be a positive integer.")
        start_date = dataframe.index[-1] - pd.DateOffset(years=years)
        dataframe = dataframe.loc[start_date:]

    # Compute the return
    return (dataframe.iloc[-1] / dataframe.iloc[0] - 1) * 100

def compute_mar(dataframe):
    """
    Function to calculate MAR (Return Over Maximum Drawdown) given a dataframe of
    prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.

    Returns
    -------
    float
        The MAR of the input data.
    """
    # Calculate the CAGR for the given data
    cagr = compute_cagr(dataframe)

    # Calculate the Maximum Drawdown for the given data
    max_drawdown = compute_drawdowns(dataframe).min().abs()

    # Calculate the MAR by dividing the CAGR by the absolute value of the Maximum
    # Drawdown
    mar = cagr.div(max_drawdown)

    return mar

def filter_by_date(dataframe, years=0):
    """
    Filters a DataFrame by date, keeping only the specified number of years of data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    years : int, optional
        Number of years to keep in the filtered DataFrame. Default is 0, which means no
        filtering.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only the specified number of years of data.
    """

    if years == 0:
        return dataframe

    last_date = dataframe.index[-1]
    start_date = last_date - pd.DateOffset(years=years)

    # Handle the case when the start_date is a leap year (Feb 29)
    if start_date.month == 2 and start_date.day == 29:
        start_date = start_date - pd.DateOffset(days=1)

    # Select the data within the specified range
    filtered_dataframe = dataframe[start_date:]
    
    return filtered_dataframe

def compute_cagr(dataframe, years='', decimals=2):
    """
    Function to calculate CAGR given a dataframe of prices

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    years : int or 'ytd', optional
        Number of years to filter the data before computing the CAGR, or 'ytd'
        for year-to-date.
    decimals : int, optional
        Number of decimal places to round the result. Default is 2.

    Returns
    -------
    float or pandas.Series
        The CAGR for each column in the input DataFrame, rounded to the specified
        number of decimals.
    """
    if years == 'ytd':
        last_year = date.today().year - 1
        last_year_end = dataframe.loc[str(last_year)].iloc[-1].name
        dataframe = dataframe[last_year_end:]
        hpr = dataframe.iloc[-1][0] / dataframe.iloc[0][0] - 1
        ytd_return = hpr * 100
        return round(ytd_return, decimals)
    elif isinstance(years, int):
        d1 = dataframe.index[0]
        d2 = dataframe.index[-1]
        delta = d2 - d1
        days = delta.days
        years = years
        if days > years * 364:
            dataframe = filter_by_date(dataframe, years=years)
            value = (dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)
            return round(value, decimals)
        else:
            return str('-')
    else:
        years = len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq='D')) / 365.25
        return (dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)

def compute_sharpe(dataframe, years='', freq='daily', risk_free_rate=0):
    """
    Function to calculate the Sharpe ratio given a dataframe of prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    years : int or empty string, optional
        The number of years to include in the calculation. Default is an empty string,
        which means the function uses the entire DataFrame.
    freq : str, optional
        Frequency of the data. Possible values: 'daily', 'monthly', 'quarterly', 'auto'.
        Default is 'daily'.
    risk_free_rate : float, optional
        The risk-free rate of return. Default is 0.

    Returns
    -------
    pandas.Series
        A pandas Series containing the Sharpe ratio for each column in the input
        DataFrame.
    """
    excess_return = compute_cagr(dataframe, years) - risk_free_rate
    return excess_return.div(compute_std_dev(dataframe, freq))

def compute_max_dd(dataframe):
    """
    Compute the maximum drawdown of a given DataFrame of prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.

    Returns
    -------
    pandas.Series
        A pandas Series containing the maximum drawdown for each column in the input
        DataFrame.
    """
    return compute_drawdowns(dataframe).min()

def print_title(title_str):
    print("\n" + title_str + "\n" + "-" * len(title_str))
    
def compute_performance_table(
    dataframe, years="si", freq="daily", numeric=False, ms_table=False, title=True):
    """
    Function to calculate a performance table given a dataframe of prices.
    Takes into account the frequency of the data.
    """

    if years == "si":
        years = (
            len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq="D"))
            / 365.25
        )

        metrics = pd.DataFrame(
            [
                compute_cagr(dataframe, years),
                compute_std_dev(dataframe, freq),
                compute_sharpe(dataframe, years, freq),
                compute_max_dd(dataframe),
                compute_mar(dataframe),
            ]
        )

        metrics.index = ["CAGR", "StdDev", "Sharpe", "Max DD", "MAR"]

        metrics = round(metrics.transpose(), 2)

        # Format as percentage
        if not numeric:
            metrics["CAGR"] = (metrics["CAGR"] / 100).apply("{:.2%}".format)
            metrics["StdDev"] = (metrics["StdDev"] / 100).apply("{:.2%}".format)
            metrics["Max DD"] = (metrics["Max DD"] / 100).apply("{:.2%}".format)

        if title:
            start = str(dataframe.index[0])[0:10]
            end = str(dataframe.index[-1])[0:10]
            print_title("Performance from " + start + " to " + end +
                        " (≈ "+ str(round(years, 1))+ " years)")

        return metrics

    else:
        dataframe = filter_by_date(dataframe, years)
        metrics = pd.DataFrame(
            [
                compute_cagr(dataframe, years=years),
                compute_std_dev(dataframe),
                compute_sharpe(dataframe),
                compute_max_dd(dataframe),
                compute_mar(dataframe),
            ]
        )
        metrics.index = ["CAGR", "StdDev", "Sharpe", "Max DD", "MAR"]

        metrics = round(metrics.transpose(), 2)

        # Format as percentage
        if not numeric:
            metrics["CAGR"] = (metrics["CAGR"] / 100).apply("{:.2%}".format)
            metrics["StdDev"] = (metrics["StdDev"] / 100).apply("{:.2%}".format)
            metrics["Max DD"] = (metrics["Max DD"] / 100).apply("{:.2%}".format)

        if title:
            start = str(dataframe.index[0])[0:10]
            end = str(dataframe.index[-1])[0:10]

            if years == 1:
                print_title("Performance from " + start+ " to " + 
                            end + " ("+ str(years)+ " year)")
            else:
                print_title("Performance from " + start + " to " +
                end+ " ("+ str(years)+ " years)")

        return metrics

def compute_drawdowns_i(dataframe):
    '''
    Function to compute drawdowns based on 
    the initial value of a time series
    given a DataFrame of prices
    '''
    if not isinstance(dataframe, pd.DataFrame) or not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError("Input must be a pandas DataFrame with a DateTimeIndex")

    drawdowns = (dataframe / dataframe.iloc[0] -1 ) * 100
    drawdowns = drawdowns[drawdowns < 0].fillna(0)
    

    return drawdowns

def compute_time_period(timestamp_1, timestamp_2):
    """
    Function to compute the time difference between two timestamps in years, months,
    and days.

    Parameters
    ----------
    timestamp_1 : datetime.datetime
        The first timestamp.
    timestamp_2 : datetime.datetime
        The second timestamp.

    Returns
    -------
    str
        A string representing the time difference in years, months, and days.
    """

    # Calculate the difference in years, months, and days
    year = timestamp_1.year - timestamp_2.year
    month = timestamp_1.month - timestamp_2.month
    day = timestamp_1.day - timestamp_2.day

    # If the month difference is negative, adjust the year and month
    if month < 0:
        year = year - 1
        month = 12 + month

    # If the day difference is negative, adjust the day
    if day < 0:
        day = -day

    # Return a string representation of the time difference in years, months, and days
    return f'{year} Years {month} Months {day} Days'

def compute_drawdowns_periods(df):
    """
    Function to compute the drawdown periods (time duration between maximum points) 
    for a given DataFrame of drawdowns (where drawdown == 0).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing maximum points in drawdowns (where drawdown == 0).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the drawdown periods between maximum points.
    """
    
    # Initialize an empty list to store drawdown periods
    drawdown_periods = list()

    # Loop through the index of the input DataFrame
    for i in range(0, len(df.index)):
      
        # Compute the time period between the current index and the previous one
        # using the compute_time_period() function (assumed to be defined)
        drawdown_periods.append(compute_time_period(df.index[i], df.index[i - 1]))
    
    # Convert the list of drawdown periods into a DataFrame
    drawdown_periods = pd.DataFrame(drawdown_periods)
    
    # Return the DataFrame containing drawdown periods
    return drawdown_periods

def compute_max_drawdown_in_period(prices, timestamp_1, timestamp_2):
    """
    Function to compute the maximum drawdown within a specified time period
    for a given DataFrame of prices.
    
    Parameters
    ----------
    prices : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    timestamp_1 : str, datetime.datetime
        Starting timestamp of the time period for which the maximum drawdown
        should be calculated.
    timestamp_2 : str, datetime.datetime
        Ending timestamp of the time period for which the maximum drawdown
        should be calculated.

    Returns
    -------
    float
        Maximum drawdown within the specified time period.
    """
    
    # Slice the input DataFrame to include only the data within the specified time
    # period
    df = prices[timestamp_1:timestamp_2]
    
    # Compute the maximum drawdown using the compute_max_dd() function
    # (assumed to be defined)
    max_dd = compute_max_dd(df)
    
    # Return the maximum drawdown value
    return max_dd


def compute_drawdowns_min(df, prices):
    """
    Function to compute the minimum points in drawdowns for a given DataFrame
    of prices and another DataFrame of maximum points in drawdowns
    (where drawdowns == 0).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing maximum points in drawdowns (where drawdowns == 0).
        Index should be of datetime type.
    prices : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the minimum points in drawdowns.
    """
    
    # Initialize an empty list to store the minimum drawdowns
    drawdowns_min = list()

    # Iterate through the indices of the input DataFrame of maximum points in drawdowns
    for i in range(0, len(df.index) - 1):
      
        # Compute the minimum drawdown within the period between two consecutive
        # maximum points in drawdowns using the compute_max_drawdown_in_period() function
        drawdowns_min.append(compute_max_drawdown_in_period(prices, df.index[i], df.index[i + 1]))
    
    # Convert the list of minimum drawdowns to a pandas DataFrame
    drawdowns_min = pd.DataFrame(drawdowns_min)
    
    # Return the DataFrame containing the minimum drawdowns
    return drawdowns_min

def compute_drawdowns_table(prices, number=5):
    """
    Function to compute the drawdowns table for a given DataFrame of prices.

    Parameters
    ----------
    prices : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    number : int, optional
        Number of top drawdowns to display in the resulting table. Default is 5.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the top drawdowns table.
    """

    # Check if the input is a DataFrame with a DatetimeIndex
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Input must be a DataFrame with a DatetimeIndex.")
    
    # Compute drawdowns using the compute_drawdowns() function
    dd = compute_drawdowns(prices)
    
    # Find the maximum points in drawdowns (where drawdowns == 0)
    max_points = dd[dd == 0].dropna()
        
    # Initialize a DataFrame with a single row and the same index as the last date in
    # prices
    data = [0.0]
    new_data = pd.DataFrame(data, columns=['New_data'])
    new_data['Date'] = prices.index.max()
    new_data.set_index('Date', inplace=True)
    
    # Combine the max_points DataFrame with the new_data DataFrame
    max_points = max_points.loc[~max_points.index.duplicated(keep='first')]
    max_points = pd.DataFrame(pd.concat([max_points, new_data], axis=1).iloc[:, 0])
    
    # Compute the drawdown periods using the compute_drawdowns_periods() function
    dp = compute_drawdowns_periods(max_points)
    dp.set_index(max_points.index, inplace=True)
    
    # Concatenate the max_points and drawdown periods DataFrames
    df = pd.concat([max_points, dp], axis=1)
    
    # Preprocess the resulting DataFrame
    df.index.name = 'Date'
    df.reset_index(inplace=True)
    df['End'] = df['Date'].shift(-1)
    df[0] = df[0].shift(-1)
    
    # Compute the minimum points in drawdowns using the compute_drawdowns_min() function
    df['values'] = round(compute_drawdowns_min(max_points, prices), 2)
    
    # Sort the DataFrame by drawdown values and assign numbers to each row
    df = df.sort_values(by='values')
    df['Number'] = range(1, len(df) + 1)
    
    # Reset the index and rename the columns
    df.reset_index(inplace=True)
    df.columns = ['index', 'Begin', 'point', 'Length', 'End', 'Depth', 'Number']
    df = df[['Begin', 'End', 'Depth', 'Length']].head(number)
    
    # Format the Depth column as percentages
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: str(x) + '%')
    
    # Set the index to numbers and format the Begin and End columns as strings
    df.set_index(np.arange(1, number + 1), inplace=True)
    df['End'] = df['End'].astype(str)
    df['Begin'] = df['Begin'].astype(str)

    # Replace the last date in the End column with 'N/A'
    for i in range(0, len(df['End'])):
        if df['End'].iloc[i] == str(prices.iloc[-1].name)[0:10]:
            df['End'].iloc[i] = 'N/A'

    # Return the DataFrame containing the top drawdowns table
    return df

def merge_time_series(df_1, df_2, on='', how='outer'):
    """
    Merge two pandas DataFrames with datetime index on a specified column or index.

    Parameters
    ----------
    df_1 : pandas.DataFrame
        The first DataFrame to be merged. The index should be of datetime type.
    df_2 : pandas.DataFrame
        The second DataFrame to be merged. The index should be of datetime type.
    on : str, optional
        The column to merge the DataFrames on. If not provided or an empty string,
        the function will merge on the index. Default is an empty string.
    how : {'left', 'right', 'outer', 'inner'}, default 'outer'
        The type of merge to be performed:
            - 'left': use only keys from left frame, similar to a SQL left outer
                      join; preserve key order.
            - 'right': use only keys from right frame, similar to a SQL right
                       outer join; preserve key order.
            - 'outer': use union of keys from both frames, similar to a SQL full
                       outer join; sort keys lexicographically.
            - 'inner': use intersection of keys from both frames, similar to a SQL
                       inner join; preserve the order of the left keys.

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame containing the data from both input DataFrames, merged
        based on the specified column or index.
    """

    df = df_1.merge(df_2, how=how, left_index=True, right_index=True)
    return df

lightcolors = [
    'royalblue',
   'rgb(111, 231, 219)',
   'rgb(131, 90, 241)',
              
              ] * 10

colors_list=['royalblue', 'darkorange',
           'dimgrey', 'rgb(86, 53, 171)',  'rgb(44, 160, 44)',
           'rgb(214, 39, 40)', '#ffd166', '#62959c', '#b5179e',
           'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
           'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
           'rgb(188, 189, 34)', 'rgb(23, 190, 207)'] * 10

def ichart(data, title='', colors=colors_list, yTitle='', xTitle='', style='normal',
           width=990, height=500, hovermode='x', yticksuffix='', ytickprefix='',
           ytickformat="", source_text='', y_position_source=-0.125, xticksuffix='',
           xtickprefix='', xtickformat="", dd_range=[-50, 0], y_axis_range=None,
           log_y=False, image='', size='fixed'):
    """
    Create an interactive chart using Plotly.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing data to be plotted.
    title : str, optional
        Title of the chart. Default is an empty string.
    colors : list, optional
        List of colors    
    style:
        normal, area, drawdowns_histogram    
    colors:
        color_list or lightcolors
    hovermode:
        'x', 'x unified', 'closest'
    y_position_source:
        -0.125 or bellow
    dd_range:
        [-50, 0]
    ytickformat:
        ".1%"
    image:
        'forum' ou 'fp'
    size : str, optional
        Chart size mode. 'fixed' for fixed width and height, 'auto' to let the chart
        auto adjust its size. Default is 'fixed'.
    """

    if image == 'fp':
        image = 'https://raw.githubusercontent.com/LuisSousaSilva/Articles_and_studies/master/FP-cor-positivo.png'

    fig = go.Figure()

    # Update layout options
    layout_options = dict(
        paper_bgcolor='#F5F6F9',
        plot_bgcolor='#F5F6F9', 
        hovermode=hovermode,
        title=title,
        title_x=0.5,
        yaxis = dict(
            ticksuffix=yticksuffix,
            tickprefix=ytickprefix,
            tickfont=dict(color='#4D5663'),
            gridcolor='#E1E5ED',
            range=y_axis_range,
            titlefont=dict(color='#4D5663'),
            zerolinecolor='#E1E5ED',
            title=yTitle,
            showgrid=True,
            tickformat=ytickformat,
        ),
        xaxis = dict(
            title=xTitle,
            tickfont=dict(color='#4D5663'),
            gridcolor='#E1E5ED',
            titlefont=dict(color='#4D5663'),
            zerolinecolor='#E1E5ED',
            showgrid=True,
            tickformat=xtickformat,
            ticksuffix=xticksuffix,
            tickprefix=xtickprefix,
        ),
        images= [dict(
            name= "watermark_1",
            source= image,
            xref= "paper",
            yref= "paper",
            x= -0.05500,
            y= 1.250,
            sizey= 0.20,
            sizex= 0.20,
            opacity= 1,
            layer= "below"
        )],
        annotations=[dict(
            xref="paper",
            yref="paper",
            x= 0.5,
            y= y_position_source,
            xanchor="center",
            yanchor="top",
            text=source_text,
            showarrow= False,
            font= dict(
                family="Arial",
                size=12,
                color="rgb(150,150,150)"
            )
        )]
    )

    # Set width and height only if size is set to 'fixed'
    if size == 'fixed':
        layout_options.update(dict(width=width, height=height))

    fig.update_layout(**layout_options)

    if log_y:
        fig.update_yaxes(type="log")

    if style == 'normal':
        z = -1

        for i in data:
            z = z + 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[i],
                mode='lines',
                name=i,
                line=dict(width=1.3,
                          color=colors[z]),
            ))

    if style == 'area':
        z = -1

        for i in data:
            z = z + 1
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[i],
                hoverinfo='x+y',
                mode='lines',
                name=i,
                line=dict(width=0.7,
                          color=colors[z]),
                stackgroup='one'  # define stack group
            ))

    if style == 'drawdowns_histogram':
        fig.add_trace(go.Histogram(x=data.iloc[:, 0],
                                   histnorm='probability',
                                   marker=dict(colorscale='RdBu',
                                               reversescale=False,
                                               cmin=-24,
                                               cmax=0,
                                               color=np.arange(start=dd_range[0],
                                                                stop=dd_range[1]),
                                               line=dict(color='white', width=0.2)),
                                   opacity=0.75,
                                   cumulative=dict(enabled=True)))

    return fig

def color_negative_red(value):
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value < 0:
    color = 'red'
  elif value > 0:
    color = 'green'
  else:
    color = 'black'

  return 'color: %s' % color

def compute_yearly_returns(dataframe, start='1900', end='2100', style='table',
                        title='Yearly Returns', color=False, warning=True,
                        numeric=False): 
    '''
    Style: table // string // chart
    '''

    dataframe = dataframe.ffill().dropna()

    # Getting start date
    start = str(dataframe.index[0])[0:10]

    # Resampling to yearly (business year)
    yearly_quotes = dataframe.resample('BA').last()

    # Adding first quote (only if start is in the middle of the year)
    yearly_quotes = pd.concat([dataframe.iloc[:1], yearly_quotes])
    first_year = dataframe.index[0].year - 1
    last_year = dataframe.index[-1].year + 1

    # Returns
    yearly_returns = ((yearly_quotes / yearly_quotes.shift(1)) - 1) * 100
    yearly_returns = yearly_returns.set_index([list(range(first_year, last_year))])

    #### Inverter o sentido das rows no dataframe ####
    yearly_returns = yearly_returns.loc[first_year + 1:last_year].transpose()
    yearly_returns = round(yearly_returns, 2)

    # As strings and percentages
    yearly_returns.columns = yearly_returns.columns.map(str)    
    yearly_returns_numeric = yearly_returns.copy()

    if style=='table' and color==False:
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.format("{:.2%}")
        print_title(title)

    elif style=='numeric':
        yearly_returns = yearly_returns_numeric.copy()
    
    elif style=='table':
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.applymap(color_negative_red).format("{:.2%}")
        print_title(title)

    elif style=='numeric':
        yearly_returns = yearly_returns_numeric.copy()

    elif style=='string':
        for column in yearly_returns:
            yearly_returns[column] = yearly_returns[column].apply( lambda x : str(x) + '%')

    elif style=='chart':
        fig, ax = plt.subplots()
        fig.set_size_inches(yearly_returns_numeric.shape[1] * 1.25, yearly_returns_numeric.shape[0] + 0.5)
        yearly_returns = sns.heatmap(yearly_returns_numeric, annot=True, cmap="RdYlGn", linewidths=.2, fmt=".2f", cbar=False, center=0)
        for t in yearly_returns.texts: t.set_text(t.get_text() + "%")
        plt.title(title)
    
    else:
        print('At least one parameter has a wrong input')

    return yearly_returns

def compute_time_series(dataframe: pd.DataFrame, start_value: float = 100) -> pd.DataFrame:
    """
    Compute the growth time series given a DataFrame of returns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing return data.
    start_value : float, optional
        Initial value of the time series. Default is 100.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the growth time series for each column in the input DataFrame.
    """
    # Validate input
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input 'dataframe' must be a pandas DataFrame.")

    # Calculate growth time series
    return (np.exp(np.log1p(dataframe).cumsum())) * start_value