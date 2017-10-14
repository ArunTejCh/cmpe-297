import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

dates = pa.date_range('20170912', periods=10)

df = pa.DataFrame(np.random.randn(10,4), index=dates, columns=list('ABCD'))


print df.describe()