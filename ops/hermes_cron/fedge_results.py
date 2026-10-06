import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fedge_paper import main
sys.exit(main("results"))
