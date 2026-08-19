import pytest

from shelfsense.models import Product, ShelfSlot


@pytest.fixture
def product() -> Product:
    return Product("SKU1", "Oat Milk", 2.50)


@pytest.fixture
def slot(product: Product) -> ShelfSlot:
    return ShelfSlot(product=product, quantity=4, capacity=10)
