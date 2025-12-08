"""empty message

Revision ID: 4a672d34c5fc
Revises: 013_add_recommendation_feedback_table, 9503d3a1c573
Create Date: 2025-12-07 20:15:45.340141

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a672d34c5fc'
down_revision: Union[str, Sequence[str], None] = ('013_add_recommendation_feedback_table', '9503d3a1c573')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
