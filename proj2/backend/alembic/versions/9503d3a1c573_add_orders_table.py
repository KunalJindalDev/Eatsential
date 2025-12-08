"""add_orders_table

Revision ID: 9503d3a1c573
Revises: 012_add_menu_item_allergens_association
Create Date: 2025-12-06 13:52:16.501675

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9503d3a1c573'
down_revision: Union[str, Sequence[str], None] = '012_add_menu_item_allergens_association'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
