import json
from pathlib import Path

import click
import pandas as pd

from hospital_sim.simulation import HospitalSim, load_config


@click.group()
def cli():
    """Hospital Simulation CLI"""


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True, help="YAML config path")
@click.option("--outdir", type=click.Path(file_okay=False), default="outputs", help="Directory to write outputs")
@click.option("--csv", "save_csv", is_flag=True, default=True, help="Save detailed patient CSV")
@click.option("--kpi", "save_kpi", is_flag=True, default=True, help="Save KPI JSON")
@click.option("--prefix", default="run", help="Filename prefix for outputs")
def run(config_path, outdir, save_csv, save_kpi, prefix):
    cfg = load_config(config_path)
    sim = HospitalSim(cfg)
    df = sim.run()

    kpis = sim.kpis(df)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if save_csv:
        df.to_csv(outdir / f"{prefix}_patients.csv", index=False)

    if save_kpi:
        with open(outdir / f"{prefix}_kpis.json", "w", encoding="utf-8") as f:
            json.dump(kpis, f, indent=2)

    click.echo("Run complete.")
    click.echo(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    cli()
