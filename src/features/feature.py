# TODO add features engineering
def feature_engineering(data):
    data['TimesBeforeLast'] = data.apply(lambda x: x['Times'] - x['Recency'], axis=1)
    data['FrequencyBeforeLast'] = data.apply(lambda x: (x['Times'] - x['Recency']) / x['Frequency'], axis=1)
    return data

