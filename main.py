from datetime import datetime

import pandas as pd
import os
import logging
import sys
import click
import torch
import warnings
import backbones
import glass
import utils


@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
def main(**kwargs):
    pass


@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=640)
@click.option("--eval_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--mining", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--radius", type=float, default=0.75)
@click.option("--p", type=float, default=0.5)
@click.option("--lr", type=float, default=0.0001)
@click.option("--svd", type=int, default=0)
@click.option("--step", type=int, default=20)
@click.option("--limit", type=int, default=392)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        train_backbone,
        pre_proj,
        mining,
        noise,
        radius,
        p,
        lr,
        svd,
        step,
        limit,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape, device):
        glasses = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            glass_inst = glass.GLASS(device)
            glass_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
            )
            glasses.append(glass_inst.to(device))
        return glasses

    return "get_glass", get_glass


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("aug_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=1, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--downsampling", default=8, type=int)
@click.option("--augment", is_flag=True)
def dataset(
        name,
        data_path,
        aug_path,
        subdatasets,
        batch_size,
        resize,
        imagesize,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        downsampling,
        augment,
):
    _DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"], "visa": ["datasets.visa", "VisADataset"],
                 "mpdd": ["datasets.mvtec", "MVTecDataset"], "wfdd": ["datasets.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, test, get_name=name):
        dataloaders = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader.name = get_name + "_" + subdataset

            if test == 'ckpt':
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,
                    rand_aug=rand_aug,
                    downsampling=downsampling,
                    augment=augment,
                    batch_size=batch_size,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )

                train_dataloader.name = test_dataloader.name
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}")
            else:
                train_dataloader = test_dataloader
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={0} test={len(test_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders


@main.result_callback()
def run(
        methods,
        results_path,
        gpu,
        seed,
        log_group,
        log_project,
        run_name,
        test,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    result_collect = []
    data = {'Class': [], 'Distribution': [], 'Foreground': []}
    df = pd.DataFrame(data)
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        glass_list = methods["get_glass"](imagesize, device)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, GLASS in enumerate(glass_list):
            flag = 0., 0., 0., 0., 0., -1.
            if GLASS.backbone.seed is not None:
                utils.fix_seeds(GLASS.backbone.seed, device)

            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            if test == 'ckpt':
                flag = GLASS.trainer(dataloaders["training"], dataloaders["testing"], dataset_name)
                if type(flag) == int:
                    row_dist = {'Class': dataloaders["training"].name, 'Distribution': flag, 'Foreground': flag}
                    df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])

            if type(flag) != int:
                i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = GLASS.tester(dataloaders["testing"], dataset_name)
                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "image_auroc": i_auroc,
                        "image_ap": i_ap,
                        "pixel_auroc": p_auroc,
                        "pixel_ap": p_ap,
                        "pixel_pro": p_pro,
                        "best_epoch": epoch,
                    }
                )

                if epoch > -1:
                    for key, item in result_collect[-1].items():
                        if isinstance(item, str):
                            continue
                        elif isinstance(item, int):
                            print(f"{key}:{item}")
                        else:
                            print(f"{key}:{round(item * 100, 2)} ", end="")

                # save results csv after each category
                print("\n")
                result_metric_names = list(result_collect[-1].keys())[1:]
                result_dataset_names = [results["dataset_name"] for results in result_collect]
                result_scores = [list(results.values())[1:] for results in result_collect]
                utils.compute_and_store_final_results(
                    run_save_path,
                    result_scores,
                    result_metric_names,
                    row_names=result_dataset_names,
                )

    # save distribution judgment xlsx after all categories
    if len(df['Class']) != 0:
        os.makedirs('./datasets/excel', exist_ok=True)
        xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
        df.to_excel(xlsx_path, index=False)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
