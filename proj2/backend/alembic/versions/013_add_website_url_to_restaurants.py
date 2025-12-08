"""Add website_url column to restaurants table

Revision ID: 013_add_website_url_to_restaurants
Revises: 012_add_menu_item_allergens_association
Create Date: 2025-12-07 10:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013_add_website_url_to_restaurants"
down_revision: Union[str, Sequence[str], None] = "fded39fd92e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add website_url column."""
    op.add_column("restaurants", sa.Column("website_url", sa.String(500), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema - remove website_url column."""
    op.drop_column("restaurants", "website_url")
    # ### end Alembic commands ###
