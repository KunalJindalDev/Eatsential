"""empty message

Revision ID: f2861a6fe3c3
Revises: 013_add_website_url_to_restaurants, 9a022a31aff4
Create Date: 2025-12-08 01:16:25.045120

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2861a6fe3c3'
down_revision: Union[str, Sequence[str], None] = ('013_add_website_url_to_restaurants', '9a022a31aff4')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
