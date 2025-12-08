"""
Test Suite for Recommendation Diversity Feature
================================================

This module contains comprehensive tests for the recommendation diversity feature,
which ensures that meal recommendations come from different restaurants to provide
users with varied dining options.

## Test Coverage

### Unit Tests (test_recommendation_diversity.py)
- Basic diversity filtering with multiple restaurants
- Restaurant interleaving logic
- Single restaurant handling
- Custom max_same_restaurant parameter
- Edge cases (empty lists, None place_ids, single items)
- Score ordering preservation

### Integration Tests (test_recommendation_diversity_integration.py)
- Diversity across different cuisines (Italian, Mexican, etc.)
- Baseline vs LLM mode comparison
- Multiple cuisine filters
- Response quality and completeness
- Restaurant data validation
- Score sorting verification

## Running Tests

### Run all diversity tests:
```bash
cd proj2/backend
python3 -m pytest tests/services/test_recommendation_diversity.py tests/integration/test_recommendation_diversity_integration.py -v
```

### Run unit tests only:
```bash
python3 -m pytest tests/services/test_recommendation_diversity.py -v
```

### Run integration tests only:
```bash
python3 -m pytest tests/integration/test_recommendation_diversity_integration.py -v
```

### Run specific test:
```bash
python3 -m pytest tests/services/test_recommendation_diversity.py::TestRecommendationDiversity::test_ensure_diverse_restaurants_basic -v
```

## Test Fixtures

The tests use the following fixtures from conftest.py:
- `client`: FastAPI test client
- `rec_test_user`: Test user with authentication
- `rec_test_menu_items`: Sample menu items from different restaurants
- `rec_user_with_profile`: Test user with health profile

## Key Assertions

1. **Diversity**: Recommendations should come from at least 2 different restaurants when 3+ items are returned
2. **Limit per Restaurant**: No restaurant should have more than `max_same_restaurant` items (default: 2)
3. **Restaurant Data**: All items should have `restaurant_name`, `restaurant_place_id`, and `restaurant_address`
4. **Place ID Format**: Google Place IDs should start with "ChI" and be at least 10 characters
5. **Score Ordering**: Items should maintain score-based ranking within diversity constraints

## Example Test Output

```
tests/services/test_recommendation_diversity.py::TestRecommendationDiversity::test_ensure_diverse_restaurants_basic PASSED
tests/services/test_recommendation_diversity.py::TestRecommendationDiversity::test_ensure_diverse_restaurants_interleaving PASSED
tests/integration/test_recommendation_diversity_integration.py::TestRecommendationDiversityByCuisine::test_italian_recommendations_diversity_baseline PASSED
tests/integration/test_recommendation_diversity_integration.py::TestRecommendationDiversityByCuisine::test_mexican_recommendations_diversity_llm PASSED
```

## Maintenance Notes

- Update `max_same_restaurant` value in tests if the default changes in RecommendationService
- Add new cuisine-specific tests when expanding restaurant coverage
- Monitor test performance as database grows
"""
