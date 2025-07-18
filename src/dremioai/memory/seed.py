#
#  Copyright (C) 2017-2025 Dremio Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Seed script to populate sample business memories for demo purposes.
"""

from dremioai.memory.storage import MemoryStorage
from dremioai.log import logger
import asyncio


SAMPLE_MEMORIES = [
    {
        "id": "metric:ARR",
        "text": """ARR (Annual Recurring Revenue) Definition:

ARR is NOT simply MRR * 12. Our calculation method:

1. Take all active subscriptions as of the measurement date
2. Normalize to annual amounts:
   - Monthly subscriptions: multiply by 12
   - Quarterly subscriptions: multiply by 4
   - Annual subscriptions: use as-is
3. Include only recurring revenue (exclude one-time fees, setup costs, professional services)
4. Use the contracted amount, not the billed amount
5. For multi-year contracts, use the annual amount for the current contract year

Formula: SUM(subscription_amount * frequency_multiplier) WHERE subscription_status = 'active' AND revenue_type = 'recurring'

This ensures we capture the true recurring revenue run rate and avoid double-counting or including non-recurring items.""",
        "tags": ["metric", "finance", "revenue", "ARR", "subscription"]
    },
    {
        "id": "metric:MRR",
        "text": """MRR (Monthly Recurring Revenue) Definition:

MRR represents the predictable monthly revenue from all active subscriptions.

Calculation method:
1. Take all active subscriptions as of month-end
2. Normalize to monthly amounts:
   - Monthly subscriptions: use as-is
   - Quarterly subscriptions: divide by 3
   - Annual subscriptions: divide by 12
3. Include only recurring revenue components
4. Exclude one-time payments, setup fees, and professional services

Formula: SUM(subscription_amount / frequency_divisor) WHERE subscription_status = 'active' AND revenue_type = 'recurring'

Key difference from ARR: MRR is the monthly snapshot, while ARR projects the annual run rate. ARR ≠ MRR * 12 because they may be calculated at different points in time with different active subscription bases.""",
        "tags": ["metric", "finance", "revenue", "MRR", "subscription"]
    },
    {
        "id": "metric:ARR_vs_MRR",
        "text": """ARR vs MRR: Key Differences

While related, ARR and MRR serve different purposes:

ARR (Annual Recurring Revenue):
- Forward-looking annual run rate
- Used for annual planning and forecasting
- Calculated: Current active subscriptions normalized to annual amounts
- Timing: Point-in-time snapshot projected annually

MRR (Monthly Recurring Revenue):
- Monthly recurring revenue snapshot
- Used for monthly tracking and short-term trends
- Calculated: Current active subscriptions normalized to monthly amounts
- Timing: Month-end snapshot

IMPORTANT: ARR ≠ MRR × 12
This is because:
1. They may be calculated at different dates
2. Subscription base changes between measurements
3. Contract terms and pricing may differ
4. Churn and new acquisitions affect the base differently

Always use the specific calculation method for each metric rather than converting between them.""",
        "tags": ["metric", "finance", "revenue", "ARR", "MRR", "comparison"]
    },
    {
        "id": "data:customer_success_db",
        "text": """Customer Success Database Performance Note:

The customer success database (cs_prod.customer_data) tends to be slow during business hours due to heavy usage by the CS team for real-time customer interactions.

Recommendations:
1. Use the materialized view cs_analytics.customer_summary for reporting queries
2. For real-time data needs, query during off-peak hours (before 9 AM or after 6 PM EST)
3. Consider using the cached customer metrics in the data warehouse (dw.customer_metrics) which is refreshed every 4 hours
4. For large analytical queries, use the read replica: cs_prod_replica.customer_data

Performance tip: The customer_summary view includes pre-calculated health scores and engagement metrics, which eliminates the need for complex joins in most reporting scenarios.""",
        "tags": ["data", "performance", "customer_success", "database", "optimization"]
    },
    {
        "id": "process:monthly_revenue_close",
        "text": """Monthly Revenue Close Process:

Our monthly revenue recognition follows this sequence:

1. Day 1-3: Billing team finalizes invoices and subscription changes
2. Day 4-5: Revenue team validates recurring vs. one-time classifications
3. Day 6-7: Finance reviews and approves revenue recognition entries
4. Day 8-10: Data team updates revenue metrics (ARR, MRR, churn rates)
5. Day 11-12: Executive reporting and board metrics finalized

Key validation points:
- All subscription changes must be reflected in the billing system by day 3
- Revenue recognition must align with ASC 606 standards
- ARR and MRR calculations must reconcile with billing data
- Churn analysis includes both voluntary and involuntary churn

Data sources:
- Billing: billing_prod.subscriptions
- Revenue: finance_prod.revenue_recognition  
- Metrics: analytics.monthly_metrics

Contact: revenue-ops@company.com for questions about the close process.""",
        "tags": ["process", "finance", "revenue", "monthly_close", "billing"]
    },
    {
        "id": "glossary:churn_rate",
        "text": """Churn Rate Definition:

We calculate churn rate as the percentage of customers who cancel their subscriptions during a given period.

Monthly Churn Rate Formula:
(Number of customers who churned in month) / (Total customers at start of month) × 100

Key considerations:
1. Voluntary churn: Customer-initiated cancellations
2. Involuntary churn: Failed payments, expired cards (after retry period)
3. Downgrades are NOT counted as churn (tracked separately as contraction)
4. Upgrades within the same month don't affect churn calculation

Annual churn rate is NOT monthly churn × 12. Use cohort analysis for annual churn.

Benchmark: Our target monthly churn rate is <5% for enterprise customers, <8% for SMB customers.

Data source: analytics.churn_analysis table, updated daily.""",
        "tags": ["metric", "churn", "customer", "retention", "glossary"]
    }
]


async def seed_memories():
    """Seed the memory database with sample business memories."""
    storage = MemoryStorage()
    
    logger().info("Starting memory seeding process...")
    
    for memory_data in SAMPLE_MEMORIES:
        try:
            result = storage.put_memory(
                text=memory_data["text"],
                id=memory_data["id"],
                tags=memory_data["tags"]
            )
            logger().info(f"Seeded memory: {memory_data['id']}")
        except Exception as e:
            logger().error(f"Failed to seed memory {memory_data['id']}: {str(e)}")
    
    logger().info(f"Completed seeding {len(SAMPLE_MEMORIES)} memories")


def main():
    """Main entry point for seeding script."""
    asyncio.run(seed_memories())


if __name__ == "__main__":
    main()
