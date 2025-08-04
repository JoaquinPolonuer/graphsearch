from pathlib import Path
from dotenv import load_dotenv
from graph_types.prime import PrimeNode
from graph_types.mag import PaperNode, AuthorNode, InstitutionNode, FieldOfStudyNode
from graph_types.amazon import ProductNode, BrandNode, ColorNode, CategoryNode

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

NODE_TYPE_MAPPING = {
    "product": ProductNode,
    "brand": BrandNode,
    "color": ColorNode,
    "category": CategoryNode,
    "author": AuthorNode,
    "paper": PaperNode,
    "institution": InstitutionNode,
    "field_of_study": FieldOfStudyNode,
    "gene/protein": PrimeNode,
    "drug": PrimeNode,
    "effect/phenotype": PrimeNode,
    "disease": PrimeNode,
    "biological_process": PrimeNode,
    "molecular_function": PrimeNode,
    "cellular_component": PrimeNode,
    "exposure": PrimeNode,
    "pathway": PrimeNode,
    "anatomy": PrimeNode,
}
