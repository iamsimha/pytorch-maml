import torch
from collections import OrderedDict

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


def get_updated_params(loss, model, params, inner_update_lr):
    """
    Get the new parameters after gradient descent update. Do not modify the parameters
    Args:
        loss: Loss tensor
        params: Params use to get the loss tensor
        inner_update_lr: Learning rate in the inner loop
    Returns:
        updated_params: OrderedDict containing the new parameters 
    """
    if params is None:
        params = OrderedDict(model.named_parameters())

    grads = torch.autograd.grad(loss, params.values())
    updated_params = OrderedDict()
    for (name, param), grad in zip(params.items(), grads):
        # Gradient descent update
        updated_params[name] = param - inner_update_lr * grad
    return updated_params

def get_task_outer_loss(
    model,
    loss_fn,
    task_inner_inputs,
    task_inner_labels,
    task_outer_inputs,
    task_outer_labels,
    inner_update_lr,
    num_inner_updates,
    prefix = ""
):
    """
    Computes the MAML Loss function
    Args:
        model: torch model
        loss_fn: Loss function
        task_inner_inputs: Inputs used in the inner MAML loop
        task_inner_labels: Labels for the inner loop
        task_outer_inputs: Labels for the outer loop
        task_outer_labels: Labels for the outer loop
        inner_update_lr: Leanring rate for inner loss term
        num_inner_updates: Number of SGD updates for the inner loss function
    Returns:
        loss_outer: Outer loss obtained by using task_outer_inputs and task_outer_labels
        accuracy: Accuracy of the model on the inner inputs
    """
    inner_params = apply_inner_update(
        model,
        loss_fn,
        inner_update_lr,
        num_inner_updates,
        task_inner_inputs,
        task_inner_labels,
    )
    with torch.set_grad_enabled(model.training):
        logits_outer = model(task_outer_inputs, inner_params)
        loss_outer = loss_fn(logits_outer, task_outer_labels)
        accuracy = (torch.argmax(logits_outer, dim=-1) == task_outer_labels).float().mean().item()
    return loss_outer, accuracy
