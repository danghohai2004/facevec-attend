from src.modules.attendance.schemas import (
    DailyStatItem,
    DailyStatsResponse,
    MonthlyStatItem,
    MonthlyStatsResponse,
    SummaryDeltas,
    SummaryStatsResponse,
)


def test_summary_stats_response_shape():
    response = SummaryStatsResponse(
        total_employees=128,
        todays_attendance=94,
        average_working_hours=7.6,
        on_time_rate=91.5,
        deltas=SummaryDeltas(
            todays_attendance=2.1,
            average_working_hours=-1.3,
            on_time_rate=None,
        ),
    )

    assert response.model_dump() == {
        "total_employees": 128,
        "todays_attendance": 94,
        "average_working_hours": 7.6,
        "on_time_rate": 91.5,
        "deltas": {
            "todays_attendance": 2.1,
            "average_working_hours": -1.3,
            "on_time_rate": None,
        },
    }


def test_monthly_and_daily_stats_response_shapes():
    monthly = MonthlyStatsResponse(
        available_years=[2025, 2026],
        items=[
            MonthlyStatItem(
                month=7,
                attendance=94,
                working_hours=714.5,
                average_hours=7.6,
            )
        ],
    )
    daily = DailyStatsResponse(
        items=[DailyStatItem(day=3, average_hours=7.58)]
    )

    assert monthly.model_dump() == {
        "available_years": [2025, 2026],
        "items": [
            {
                "month": 7,
                "attendance": 94,
                "working_hours": 714.5,
                "average_hours": 7.6,
            }
        ],
    }
    assert daily.model_dump() == {
        "items": [{"day": 3, "average_hours": 7.58}]
    }
