# Check Data type

## print all object columns
df.select_dtypes(include='object').columns

## check all data types
df.dtypes
df.subscriberKey.dtypes

## check object type
type(df.subscriberKey) # return pandas.core.series.series

# Handle missing value

## slice data without missing value in a specific column, build a model, and use prediction to fill na
missing = df_sample[np.isnan(df_sample.orig_destination_distance)==False]
df['distance'] = pd.Series(dis_pred)
df.orig_destination_distance.fillna(value=df.distance, inplace=True)

## drop na with threshold
df.dropna(thresh=df.shape[0]*0.6,how='all',axis=1, inplace=True)

for i in prop.columns:
    if float(prop[i].isnull().sum())/prop.shape[0]>=0.89:
        prop.drop(i, axis=1, inplace=True)

## fill na with median/mean/..
df.fillna(df.median(), inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_value = nan, strategy='median')
imputer.fit(df)
imputer.transform(df)

## fill na with random generated values according to the distribution of available data
p = df.ethnicity.value_counts()/df.ethnicity.dropna().shape[0]
a = ['European', 'Hispanic', 'Asian', 'Other', 'African-American']
df['eth_imp'] = np.random.choice(a=a, size=df.shape[0], p=p)
df.ethnicity.fillna(value=df.eth_imp, inplace=True)

# Check data quality

## check EDD (mean, std, percentile, min, max, count, unique, freq)
df.describe(include='all')

## check dimension
df.shape

## check specific rows/columns
df.loc[1:5, 'txnNbr']
df.iloc[:, 2:4]

## check columns with single feature
single_col = [col for col in df.columns if df[col].nunique()==1]

## find indicator columns
ind_col = [col for col in df1.columns if df1[col].nunique()==2 and df1[col].max()==1 and df1[col].min()==0]

## find columns with null or inf
df.isnull().sum()
df.columns[np.isnan(df).all()]
df.columns[np.isinf(df).all()]
df.columns[np.isinf(df).any()]

## count distinct values of a column
df.majorRateplanSeg.value_counts(dropna=False)
np.unique(dbscan.labels_, return_counts=True)

## count distinct values in each group
df.groupby('txn_month_key').xtra_card_nbr.nunique()

## check length of a column
df_test.memo_events.dropna().apply(lambda x: len(x.split(","))).mean()

# Drop columns
df.drop([list_of_columns], axis=1, inplace=True)


# Rename columns
## rename all columns
df.columns = []

## rename specific columns
df.rename(columns={'default payment next month': 'default'}, inplace=True)
df.columns=df.columns.str.lower()

# Derive columns

## derive column with a function and apply
def credit(x):
    if x in ('A', 'B'):
        return "prime"
    elif x in ("C","L"):
        return 'near_prime'
    elif x in ("W","D","H","I","Y"):
        return 'sub_prime'
    elif x == 'O':
        return 'credit_O'
    else:
        return 'credit_other'

df['credit'] = df.accountCurrCreditClassCd.apply(lambda x: credit(x))

## create dummy with pandas
df1 = pd.get_dummies(df, columns=['majorRateplanSeg', 'needStateName', 'credit'], drop_first=True)

## create dummy with sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df['location_num'] = le.fit_transform(df.location)
enc = OneHotEncoder(categorical_features=[1,2,3,5,6,7,8,9,10])
X = enc.fit_transform(X)

## create feature based on multiple columns in a row
def prv_carrier(row):
    if row['prv_std_carr'] == 'UA' and row['prv_act_carr'] == 'UA':
        return 'ML'
    elif row['prv_std_carr'] == 'UA' and row['prv_act_carr'] != 'UA':
        return 'EX'
    else:
        return 'OA'
df['prv_carr'] = df.apply(prv_carrier, axis=1)

## create text features
tvec = TfidfVectorizer(stop_words='english')
tvec.fit(data.summary)
words = pd.DataFrame(tvec.transform(data.summary).todense(), columns = tvec.get_feature_names())
df = pd.concat([data, words], axis=1)

## Handle datetime

### convert to datetime type
df[i] = pd.to_datetime(df[i])

### convert to time range type
df.prv_arrv_delay = df.prv_arrv_delay.astype('timedelta64[s]')/60

### create time feature
df['prv_sch_dprt_hr'] = df.prv_sch_dprt_time.dt.hour

## create bin
df_lp['tenure_bin'] = pd.cut(df["tenure"], np.arange(0, 3*365, 30))

# Filter Data

## filter based on column value
df_test = df1[df1.reportDate == '2018-10-31']

## filter data based on index or a value range of a column
X_train_b = df_train[df_train.index.isin(list(X_train.index))]
df_lp = df.loc[df.xtra_card_nbr.isin(df[df.tenure<=30].xtra_card_nbr)]

# Check target rate
df_train.ind_Target_Acc_30D.value_counts()/df_train.shape[0]

# Standardization

## with numpy
X[col_to_std] = X[col_to_std].apply(lambda x: (x-np.nanmean(x))/np.nanstd(x))

## if you want to use same mean and std for test data
x_mean = np.nanmean(X_train[col_to_std], axis=0)
x_std = np.nanstd(X_train[col_to_std], axis=0)
X_train[col_to_std] = (X_train[col_to_std] - x_mean)/x_std

## with sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_feature = scaler.fit_transform(prop[num_col])


# Handle Outlier

## cap
def handle_outlier(x):
    low_quantiles = x.quantile(0.05)
    high_quantiles = x.quantile(0.95)

    low_outliers = (x < low_quantiles)
    high_outliers = (x > high_quantiles)
    x = x.mask(low_outliers, low_quantiles, axis=1)
    x = x.mask(high_outliers, high_quantiles, axis=1)
    return x

X = handle_outlier(X)

## drop 3 std out
df = df[np.abs(stats.zscore(df)<3).all(axis=1)]


# Drop duplicates
df.drop_duplicates(subset=['xtra_card_nbr','tenure_bin'], keep='last', inplace=True)

# Drop rows with specific values of string column
df = df[df.salary.str.contains('hour')==False]

# Sort data
df.sort_values('txnDt', ascending=True)

## feature importances
feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(30)
feature_importances.plot(kind='bar')

## sort data within a group
df.sort_values('txnDt', ascending=True).groupby('subscriberKey')

# create coef list
signal_name = list(df_train.drop(drop_list, axis=1).columns)
coef = pd.DataFrame(np.column_stack([signal_name,sgd.coef_[0]]), columns=['signal','coefficient'])
coef.coefficient = coef.coefficient.astype(float) # change data tpye
coef[coef.coefficient!=0].sort_values('coefficient',ascending=False)


# Aggregation
## Group data, concat strings
pd.DataFrame(df.groupby('subscriberKey').location.apply(lambda x: ', '.join(x))).reset_index()

## Group data based on a binned column
df.groupby(pd.cut(df["tenure"], np.arange(0, 3*365, 30))).scan_amt.mean()


# Merge table

## Join
df.merge(df1, on=['subscriberKey','txnNbr'] , how = 'inner') # left_on=, right_on=, left, right, outer

## union
pd.concat([df1, df2])

## append columns
pd.concat([df1, df4], axis=1)


# iterate through rows
for index, row in df.iterrows():
    print(row['c1'], row['c2'])


# Train Test Split

## use random
df = df.iloc[np.random.permutation(len(df))]
X_train = df[X_columns].head(500000)
X_test = df[X_columns].tail(100000)

## use sample
df.sample(frac=1)

## use sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=7)

## if x is a list
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
x_train, x_test = x_shuffled[:-50000], x_shuffled[-50000:]
y_train, y_test = y_shuffled[:-50000], y_shuffled[-50000:]

# Plotting

## Plot distribution
plt.figure(figsize=(15,8))
sns.distplot(df1.acctCLV)

## Plot x vs y
g = sns.jointplot(data=df_pred, x = 'model_pred', y = 'true_val', kind="reg")
g.fig.set_size_inches(15,15)

## pair plot variables
sns.pairplot(df, vars=df.columns[11:17], kind='scatter')
