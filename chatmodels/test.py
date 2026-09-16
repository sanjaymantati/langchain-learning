from huggingface_hub import HfApi

api = HfApi()

models = api.list_models(
    inference="warm",
    pipeline_tag="text-generation",
    sort="downloads",
    limit=20,
)

for model in models:
    print(model.id)