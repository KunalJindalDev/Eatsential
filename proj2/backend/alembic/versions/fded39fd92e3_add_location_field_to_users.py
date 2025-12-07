"""add location field to users

Revision ID: fded39fd92e3
Revises: 012_add_menu_item_allergens_association
Create Date: 2025-11-23 01:51:03.110166

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import JSON


# revision identifiers, used by Alembic.
revision: str = 'fded39fd92e3'
down_revision: Union[str, Sequence[str], None] = '012_add_menu_item_allergens_association'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add location column to users table
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "location",
                JSON,
                nullable=True,
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove location column from users table
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("location")
