#   TV 
#
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ESSdata_Thinkful.csv")
df.dropna(inplace=True)

YR2012=6
MEN=1
FEM=2

TOGETHER = 1
ALONE = 2

df=df[df['year'] == YR2012]

m=df[df['gndr'] == MEN]
f=df[df['gndr'] == FEM]

m1=pd.DataFrame((m['tvtot'].value_counts()))
m1.sort_index(axis=0, inplace=True, ascending=False)

f1=pd.DataFrame(f['tvtot'].value_counts())
f1.sort_index(axis=0, inplace=True, ascending=False)

plt.plot(m1, color='b',label='MEN')
plt.plot(f1, color='red',label='WOMEN')
hdr = "  Men v. Women TV Watching in 2012"
plt.legend()
plt.ylabel('Count')
plt.xlabel("TV Watching")
plt.title(hdr)
plt.show()
#
#FAIR
t=df[df['partner'] == TOGETHER]
a=df[df['partner'] == ALONE]

t=pd.DataFrame((t['pplfair'].value_counts()))
m1.sort_index(axis=0, inplace=True, ascending=False)

f1=pd.DataFrame(f['tvtot'].value_counts())
f1.sort_index(axis=0, inplace=True, ascending=False)

plt.plot(m1, color='b',label='MEN')
plt.plot(f1, color='red',label='WOMEN')
hdr = "  Men v. Women TV Watching in 2012"
plt.legend()
plt.ylabel('Count')
plt.xlabel("TV Watching")
plt.title(hdr)
plt.show()
#
#social meet
#
YR2012=6
df = pd.read_csv("ESSdata_Thinkful.csv")
df.dropna(inplace=True)

df=df[df['year'] == YR2012]

colors = ['red','blue','g','y','orange','purple']
color_index=0
countries = ['SE', 'CH', 'CZ','NO','DE', 'ES']
#TRUST
for key in countries:
    x=df[df['cntry'].isin([key])]
    z1=pd.DataFrame((x['sclmeet'].value_counts()))
    z1.sort_index(axis=0, inplace=True, ascending=False)
    plt.plot(z1, color=colors[color_index],label=key)
    color_index +=1
hdr = "  By Country Social Activity 2012"
plt.legend()
plt.ylabel('Count')
plt.xlabel("Socialability\n Denmark sample size small.")
plt.title(hdr)
plt.show()
#
#social activity
#
YR2014=7
df = pd.read_csv("ESSdata_Thinkful.csv")
df.dropna(inplace=True)

df=df[df['year'] == YR2014]

colors = ['red','blue','g','y','orange','purple']
color_index=0
countries = ['SE', 'CH', 'CZ','NO','DE', 'ES']

for key in countries:
    x=df[df['cntry'].isin([key])]
    z1=pd.DataFrame((x['sclact'].value_counts()))
    z1.sort_index(axis=0, inplace=True, ascending=False)
    plt.plot(z1, color=colors[color_index],label=key)
    color_index +=1
hdr = "  By Country Social Activity 2014"
plt.legend()
plt.ylabel('Count')
plt.xlabel("Social Frequency\n Denmark sample size small. Otherwise activity pretty equal.")
plt.title(hdr)
plt.show()


