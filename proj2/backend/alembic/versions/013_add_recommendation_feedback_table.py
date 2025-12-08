"""add_recommendation_feedback_table

Revision ID: 013_add_recommendation_feedback_table
Revises: 012_add_menu_item_allergens_association
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013_add_recommendation_feedback_table"
down_revision: Union[str, Sequence[str], None] = "012_add_menu_item_allergens_association"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "recommendation_feedback",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("item_type", sa.String(length=20), nullable=False),
        sa.Column("feedback_type", sa.String(length=20), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_recommendation_feedback_user_id"),
        "recommendation_feedback",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_recommendation_feedback_item_id"),
        "recommendation_feedback",
        ["item_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_recommendation_feedback_item_type"),
        "recommendation_feedback",
        ["item_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_recommendation_feedback_feedback_type"),
        "recommendation_feedback",
        ["feedback_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_recommendation_feedback_created_at"),
        "recommendation_feedback",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_recommendation_feedback_created_at"),
        table_name="recommendation_feedback",
    )
    op.drop_index(
        op.f("ix_recommendation_feedback_feedback_type"),
        table_name="recommendation_feedback",
    )
    op.drop_index(
        op.f("ix_recommendation_feedback_item_type"),
        table_name="recommendation_feedback",
    )
    op.drop_index(
        op.f("ix_recommendation_feedback_item_id"),
        table_name="recommendation_feedback",
    )
    op.drop_index(
        op.f("ix_recommendation_feedback_user_id"),
        table_name="recommendation_feedback",
    )
    op.drop_table("recommendation_feedback")

