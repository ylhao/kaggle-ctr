# kaggle 点击率预测

- click_rate_demo.ipynb: demo 熟悉流程


## 点击率预估
CTR 预估无论是在学术界还是在工业界都是一个很热的话题，尤其是在互联网的计算广告领域。在计算广告领域，CTR 的预估准确与否直接影响商业利润，所以各大公司都很重视 CTR 预估方面的工作。像在 BAT 这样的大平台，数据量巨大，不仅要考虑模型的精度，还要考虑模型训练的时间代价。线性模型更新快，非线性模型训练代价高。线性模型需要大量的特征工程，尤其是要做大量 cross-feature 来达到非线性效果。非线性模型模型本身具备拟合非线性的特点，所以相对于线性模型，做的特征工程会少很多。

## Data fields
- id: ad identifier
- click: 0/1 for non-click/click
- hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
- C1 -- anonymized categorical variable
- banner_pos
- site_id
- site_domain
- site_category
- app_id
- app_domain
- app_category
- device_id
- device_ip
- device_model
- device_type
- device_conn_type
- C14-C21 -- anonymized categorical variables

