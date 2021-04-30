from preprocess_ipn import preprocess_ipn_dataset
import argparse

parser = argparse.ArgumentParser(description='Run preprocess on IPN dataset.')

parser.add_argument("prefix", metavar="path_to_IPN_dataset_folder",
                    type=str, help="Path to the (outmost) IPN dataset folder.")

args = parser.parse_args()

preprocess_ipn_dataset(args.prefix, frames_dir="frames", segs_dir="segment")