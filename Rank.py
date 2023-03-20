import numpy as np
import pandas as pd

df = pd.DataFrame(data={'Animal': ['fox', 'Kangaroo', 'deer', 'spider', 'snake'],
                        'Number_legs': [4, 2, 4, 8, np.nan]})
print(df)

df['default_rank'] = df['Number_legs'].rank()
df['min_rank'] = df['Number_legs'].rank(method='min')
df['max_rank'] = df['Number_legs'].rank(method='max')
df['dense_rank'] = df['Number_legs'].rank(method='dense')
df['NA_keep'] = df['Number_legs'].rank(na_option='keep')
df['NA_top'] = df['Number_legs'].rank(na_option='top')
df['NA_bottom'] = df['Number_legs'].rank(na_option='bottom')
df['pct_rank'] = df['Number_legs'].rank(pct=True)
print(df)