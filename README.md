# pytorch-maml
This is pedagogical implementation of MAML Algorithm.


To run MAML on Omnliglot data do

```
sh train_maml.sh
```
train_maml.sh runs

```shell
python run_maml.py \
    --data-folder <path to omniglot data> \
    --num-classes 5 \
    --num-meta-test-classes 5 \
    --num-samples-per-class 1 \
    --num-meta-test-samples-per-class 1 \
    --batch-size 32 \
    --inner-update-lr 0.4 \
    --meta-lr 0.001 \
    --num-meta-train-iterations 2000 \
    --num-meta-test-iterations 200 \
    --num-meta-validation-iterations 4 \
    --num-inner-updates 1 \
    --meta-test-num-inner-updates 1 \
    --dim-hidden 16 \
    --validation-frequency 100 \
    --device cpu
```


The below plot, compares validation accuracy with and without MAML on 1-Shot, 5-way classification.
<p align="center">
  <img width="460" height="300" src="https://github.com/iamsimha/pytorch-maml/blob/master/logs/maml.png">
</p>

### Implementation details:

The core idea is to have pytorch modules, which can take parameters in the forward function. This code is inspired from https://github.com/tristandeleu/pytorch-meta

```python
class MetaLinear(nn.Linear):
    def forward(self, input, params):
        if params is None:
            params = OrderedDict(self.named_parameters())
            weight = params.get("weight", None)
            bias = params.get("bias", None)
        else:
            weight = params.get(self.module_prefix + ".weight", None)
            bias = params.get(self.module_prefix + ".bias", None)

        return F.linear(input, weight, bias)
```
These modules help us to easily keep track of model parameters and meta parameters.

#### Inner loop of MAML
```python
def apply_inner_update(
    model, loss_fn, inner_update_lr, num_inner_updates, inps, labels
):
    """
    Perform inner gradient update 'num_inner_updates' time. The whole batch is composed of a single task
    Args:
        model: Model
        loss_fn: Loss function
        inner_update_lr: Learning rate for the inner update
        num_inner_updates: An integer specifying the number of times,
                        inner gradient descent update should be done
        inps: Input for the inner gradient loop. Inner update loss has to be calculated on this data
                Size = [batch_size, 1, 28, 28]. 28 X 28 is the width and height of omniglot data
        labels: Labels corresponding to inpa
                Size = [batch_size,]
    Returns:
        params: Updated params of the model after `num_inner_updates`
    """
    # Start from the intial paramters of the model.
    params = None
    for step in range(num_inner_updates):
        # Do one step of inner gradient update
        # each gradient update is done with the update parameter
        model.zero_grad()
        logits = model(inps, params)
        loss = loss_fn(logits, labels)

        updated_params = get_updated_params(
            loss, model, params, inner_update_lr
        )
        # Next iteration uses new value of params
        params = updated_params
    return params
```

# Issues
Please use issue tracker to raise any issues.
