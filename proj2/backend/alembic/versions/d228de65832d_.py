"""empty message

Revision ID: d228de65832d
Revises: 013_add_recommendation_feedback_table, 9503d3a1c573
Create Date: 2025-12-07 20:46:30.247047

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd228de65832d'
down_revision: Union[str, Sequence[str], None] = ('013_add_recommendation_feedback_table', '9503d3a1c573')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
