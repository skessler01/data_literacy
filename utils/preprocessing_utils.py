import pandas as pd

# =====================================================
# Helper functions for preprocessing time series data
# =====================================================

def detect_intervals_with_missing_data(df, column='count', mode='zeros'):
    """
    Detect continuous intervals where `column` is NaN or 0.
    Returns a DataFrame with start, end, n_points.
    """
    if mode == 'zeros':
        mask = df[column] == 0
    elif mode == 'missing':
        mask = df[column].isna()
    else:
        raise ValueError("mode must be 'zeros' or 'missing'")

    grp = (mask != mask.shift()).cumsum()

    intervals = (
        df[mask]
        .assign(group=grp)
        .groupby('group')
        .agg(
            start=(column, lambda x: x.index.min()),
            end=(column, lambda x: x.index.max()),
            n_points=(column, 'size')
        )
        .reset_index(drop=True)
    )

    return intervals


def remove_long_zero_intervals(df_counter, long_zero_limit=168):
    """
    Remove intervals with more than long_zero_limit consecutive 0 counts.
    Returns cleaned df and count of removed rows.
    """
    zero_intervals = detect_intervals_with_missing_data(
        df_counter.set_index('timestamp'),
        column='count',
        mode='zeros'
    )

    intervals_to_remove = zero_intervals[
        zero_intervals['n_points'] > long_zero_limit
    ]

    exclude_mask = pd.Series(False, index=df_counter.index)

    for _, row in intervals_to_remove.iterrows():
        exclude_mask |= (
            (df_counter['timestamp'] >= row['start']) &
            (df_counter['timestamp'] <= row['end'])
        )

    removed_count = exclude_mask.sum()
    df_cleaned = df_counter.loc[~exclude_mask].copy()

    return df_cleaned, removed_count


def interpolate_short_gaps(df, missing_intervals, threshold):
    """
    Interpolate gaps with <= threshold missing points.
    Returns interpolated df and number of interpolated points.
    """
    to_interpolate = missing_intervals[
        missing_intervals['n_points'] <= threshold
    ]

    interpolate_mask = pd.Series(False, index=df.index)

    for _, row in to_interpolate.iterrows():
        interpolate_mask |= (
            (df.index >= row['start']) &
            (df.index <= row['end'])
        )

    df.loc[interpolate_mask, 'count'] = (
        df['count'].interpolate(method='time')
    )

    return df, interpolate_mask.sum()


def split_long_gaps(df, long_gaps, counter_col='counter_site'):
    """
    Split dataframe into multiple counters around long gaps.
    """
    if long_gaps.empty:
        return df

    counter_base = df[counter_col].iloc[0]
    segments = []
    last_end = df.index.min()

    for _, row in long_gaps.iterrows():
        segment = df.loc[
            last_end : row['start'] - pd.Timedelta(hours=1)
        ].copy()
        if not segment.empty:
            segments.append(segment)
        last_end = row['end'] + pd.Timedelta(hours=1)

    tail = df.loc[last_end:].copy()
    if not tail.empty:
        segments.append(tail)

    processed = []
    for i, seg in enumerate(segments):
        seg = seg.copy()
        seg[counter_col] = (
            counter_base if i == 0 else f"{counter_base}_{i+1}"
        )
        processed.append(seg)

    return pd.concat(processed).sort_index()


def cap_counter_by_time_pattern(counter_data, values_where="count", cap_const=5):
    """
    Cap outliers in counter data based on hour-of-day and day-of-year patterns.
    (quantiles and IQR calculated for each hour and day_of_year combination)

    """
    counter_data = counter_data.copy()
    
    # Extract time features
    counter_data['hour'] = counter_data['timestamp'].dt.hour
    counter_data['day_of_year'] = counter_data['timestamp'].dt.dayofyear

    # Convert to float to avoid dtype mismatch
    counter_data[values_where] = counter_data[values_where].astype(float)
    
    # Initialize capped column
    counter_data[f'{values_where}_capped'] = False
    
    # Group by hour and day_of_year and calculate IQR for each group
    for (hour, day_of_year), group_indices in counter_data.groupby(['hour', 'day_of_year']).groups.items():
        group_data = counter_data.loc[group_indices, values_where]
        
        if len(group_data) > 3:  # Need at least 4 points for meaningful quartiles
            Q1 = group_data.quantile(0.25)
            if Q1 == 0: # skip capping if Q1 is 0
                continue
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + cap_const * IQR
            
            # Mark and cap outliers
            outlier_mask = group_data > upper_bound
            counter_data.loc[group_indices[outlier_mask], f'{values_where}_capped'] = True
            counter_data.loc[group_indices[outlier_mask], values_where] = upper_bound
    
    counter_data.drop(['hour', 'day_of_year'], axis=1, inplace=True)
    return counter_data

def cap_outliers_by_time_pattern(data, values_where="count", cap_const=5):
    return data.groupby('counter_site', group_keys=False).apply(
        cap_counter_by_time_pattern, values_where=values_where, cap_const=cap_const
    )