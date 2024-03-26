import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2,
                        help="Seed for the code")
    parser.add_argument('--data_name', type=str, default="Delicious",
                        help="Name of Dataset")
    # parser.add_argument('--data_name', type=str, default="bibTex",
    #                     help="Name of Dataset")
    parser.add_argument('--train', type=float, default=0.7,
                        help="train size")  # 0.01
    parser.add_argument('--pool', type=float, default=0.2,
                        help="pool size")
    parser.add_argument('--test', type=float, default=0.1,
                        help="test size")

    # model settings
    parser.add_argument('--m_hidden', type= int, default=1024, help="Number of nodes per hidden layer")
    parser.add_argument('--m_embed', type= int, default=512, help="Number of features in embedding space")
    parser.add_argument('--m_activation', type= str, default="relu", help="Type of activation function used in model")
    parser.add_argument('--m_drop_p', type= float, default=0.1, help="Dropout ratio")

    # dataloader setting
    parser.add_argument('--batch_size', type=int, default=64, help="Seed for the code")

    # optimizer settings
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0, help="Weight decay")
    parser.add_argument('--pretrain_epochs', type=int, default=50, help="Number of weights pretraining epochs")

    args = parser.parse_args()
    return args
