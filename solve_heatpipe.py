import argparse

from configure.configure import (
    get_output_settings,
    load_config,
    make_output_dir,
    run_discretised,
    run_network,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Solve and visualise heatpipe models.")
    parser.add_argument("model", choices=["network", "discretised"], help="Which model to run.")
    parser.add_argument("--config", default="configure/config.json", help="Path to config.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_settings = get_output_settings(cfg)
    out_dir = make_output_dir(out_settings.base_dir, args.model)

    if args.model == "network":
        return run_network(cfg, out_dir)

    if args.model == "discretised":
        return run_discretised(cfg, out_dir)

    raise ValueError("Invalid function argument")


if __name__ == "__main__":
    raise SystemExit(main())
