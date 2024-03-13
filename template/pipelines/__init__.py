# {% include 'template/license_header' %}

from pipelines.evaluate import {{product_name}}_evaluation
from pipelines.feature_engineering import {{product_name}}_feature_engineering
from pipelines.finetuning import {{product_name}}_finetuning
from pipelines.merge import {{product_name}}_merging
