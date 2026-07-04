"""Add phase 1.5 models

Revision ID: 20f912b3ae3d
Revises: 
Create Date: 2026-07-04 17:37:36.145671
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20f912b3ae3d'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
