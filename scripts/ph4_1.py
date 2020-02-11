import click
from amuse.io import read_set_from_file

@click.command()
@click.option("--ic_file", default=None, help="initial condition file")
def run(ic_file):
    pass


if __name__ == '__main__':
    run()
