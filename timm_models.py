import timm
import pandas as pd
from taowei.torch2.utils.viz import print_summary
from tqdm import tqdm

res = []
model_names = sorted(timm.list_models(pretrained=True))
for i, name in tqdm(enumerate(model_names)):

	if name.startswith('ig_') or 'tf_efficientnet_b5' <= name < 'tf_efficientnet_lite':
		continue

	try:
		model = timm.create_model(name)
		cfg = model.default_cfg
		info = {'model': name}
		info.update(print_summary(model, data_shape=tuple((1,) + cfg['input_size'])))
		res.append(info)
		print(info)
	except Exception as e:
		print(e)
		pass

	if (i + 1) % 50 == 0:

		df = pd.DataFrame(res)

		df2 = pd.read_csv('results/results-imagenet.csv')
		df = df.merge(df2, how='outer')

		df.to_excel('timm_models_{}.xlsx'.format(i+1), index=False)

df = pd.DataFrame(res)

df2 = pd.read_csv('results/results-imagenet.csv')
df = df.merge(df2, how='outer')

df.to_excel('timm_models.xlsx', index=False)