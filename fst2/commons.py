import os
import re
import torch
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    SequentialSampler, 
    TensorDataset, 
    DistributedSampler)
from transformers import AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
from .utils import set_seed



def _decide_inputs(task, batch, model_type):
    if task == "ner":
        inputs = {"input_ids": batch[0], "labels": batch[3]}
        if model_type != "distilbert":
            inputs["token_type_ids"] = (
                        batch[2] if model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
    elif task == "text-classification":
        inputs = {"input_ids": batch[0],  "labels": batch[2]}
    return inputs

def _train(
    task,
    logger, 
    tb_writer, 
    model,
    tokenizer,  
    dataset, 
    max_steps, 
    num_train_epochs, 
    gradient_accumulation_steps, 
    weight_decay, 
    learning_rate,
    adam_epsilon, 
    max_grad_norm,
    warmup_steps, 
    fp16, 
    fp16_opt_level, 
    n_gpu,
    local_rank, 
    evaluate_during_training,
    evaluate_func,
    per_gpu_train_batch_size, 
    device, 
    output_dir, 
    model_type,
    model_name_or_path, 
    configs,
    seed,
    logging_steps,
    save_steps,
    **kwargs):
    """
      The basic training process function
    """
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(dataset) if local_rank == -1 else DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size)  # 直接shuffle = true， 默认使用Randomsampler

    if max_steps > 0:  # 判断所有步数
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]  
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(weight_decay),  # 默认的权重衰减值为0
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0},
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(learning_rate), eps=float(adam_epsilon))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path,
                                                                "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path,
                                                                "scheduler.pt")))

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
        * gradient_accumulation_steps
        * (torch.distributed.get_world_size() if local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from  checkpoint
    if os.path.exists(model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        if re.match("checkpoint-\d+", model_name_or_path):
            global_step = int(model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0]
    )

    set_seed(seed, n_gpu)  # Added here for reproductibility
    for _ in train_iterator:  # epoch
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):  # dataitor

                # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            # decide inputs based one task type
            inputs = _decide_inputs(task, batch, model_type)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics
                    if (
                        local_rank == -1 and evaluate_during_training
                    ):# Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate_func(mode="dev", model=model, tokenizer=tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    outputdir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(outputdir):
                        os.makedirs(outputdir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(outputdir)  # TODO: check save_pretrained method
                    tokenizer.save_pretrained(outputdir)

                    torch.save(configs, os.path.join(outputdir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", outputdir)

                    torch.save(optimizer.state_dict(), os.path.join(outputdir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(outputdir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", outputdir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def _evaluate(task,
              logger,
              model,
              model_type,
              tokenizer,
              dataset,
              labels,
              pad_token_label_id,
              per_gpu_eval_batch_size,
              n_gpu,
              local_rank,
              device,
              ouput_index):
    """
       Inference  function for evaluation during trainning or prediction.

       Args:
           task: task name, e.g ner
           logger: 
           model: model for inference
           model_type: modle type, e.g bert xlnet
           tokenizer: 
           dataset:
           labels: all classification labels
           pad_token_label_id: e.g -100 for crossentrpy loss function
           per_gpu_eval_batch_size: 
           n_gpu:
           local_rank:
           device:
           output_index:
    """
    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            inputs = _decide_inputs(task, batch, model_type)
            outputs = model(**inputs)  
            tmp_eval_loss, logits = outputs[:2] 
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=ouput_index)  

    label_map = {i: label for i, label in enumerate(labels)}
    
    # make outputs based on task category
    if task in ["ner"]:
        from seqeval.metrics import f1_score, precision_score, recall_score
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
         
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        logger.info("***** Eval results %s *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list

    elif task in ["text-classification"]:
        from sklearn.metrics import f1_score, precision_score, recall_score
        out_label_list = [None for _ in range(out_label_ids.shape[0])]
        preds_list = [None for _ in range(out_label_ids.shape[0])]
          
        for i in range(out_label_ids.shape[0]):
            if out_label_ids[i] != pad_token_label_id:
                out_label_list[i] = label_map[out_label_ids[i]]
                preds_list[i] = label_map[preds[i]]

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list, average="micro"),
            "recall": recall_score(out_label_list, preds_list, average="micro"),
            "f1": f1_score(out_label_list, preds_list, average="micro"),
        }

        logger.info("***** Eval results %s *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list


def _predict(
    task,
    evaluate_func,
    logger,
    output_dir,
    data_dir,
    result_dir,
    test_file_name, 
    delimiter,
    column_text,
    do_lower_case,
    tokenizer_class,
    model_class,
    device,
    prediction_model_dir=None
    ):
    if prediction_model_dir:
        model_path = prediction_model_dir
    else:
        # use the newest model in output_dir
        dirs = os.listdir(output_dir)
        model_dir_splits = [d.split("-") for d in dirs if re.match("^checkpoint", d)]
        latest_model_path = os.path.join(output_dir, "checkpoint-" + str(max([int(d[1]) for d in model_dir_splits])))
        model_path = latest_model_path
        tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = model_class.from_pretrained(model_path)
        model.to(device)

    result, predictions = evaluate_func(mode="test", model=model, tokenizer=tokenizer)

    # Save results
    output_test_results_file = os.path.join(result_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(result_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(data_dir, test_file_name), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    if task in ["ner"]:  
                        output_line = line.split(delimiter)[column_text] + " " + predictions[example_id].pop(0)+ "\n"
                    elif task in ["text-classification"]:
                        output_line = line.split(delimiter)[column_text].strip() + " " + predictions[example_id]+ "\n"
                    writer.write(output_line) 
                else:
                    logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


