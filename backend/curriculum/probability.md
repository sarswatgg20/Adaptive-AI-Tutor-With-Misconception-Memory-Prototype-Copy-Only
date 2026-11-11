# Probability

## Prerequisites
- Fractions and decimals
- Concept of ratios
- Basic counting and listing (for sample spaces)

## Definition
Probability measures how likely an event is to happen. It is always a number between 0 and 1 (or 0% to 100%).

- Probability of 0 = impossible (will never happen)
- Probability of 1 = certain (will always happen)
- Probability of 0.5 = equally likely to happen or not

Formula: P(event) = Number of favourable outcomes / Total number of possible outcomes

## Sample Space
The sample space is the complete list of all possible outcomes of an experiment.

Example — rolling a die: Sample space = {1, 2, 3, 4, 5, 6} → 6 total outcomes
Example — flipping a coin: Sample space = {Heads, Tails} → 2 total outcomes
Example — drawing a card from a standard deck: 52 total outcomes

## Simple Probability
Example: What is the probability of rolling a 3 on a fair die?
- Favourable outcomes: {3} → 1 outcome
- Total outcomes: 6
- P(rolling 3) = 1/6

Example: What is the probability of drawing a red card from a standard deck?
- Favourable outcomes: 26 red cards (13 hearts + 13 diamonds)
- Total outcomes: 52
- P(red card) = 26/52 = 1/2

Example: A bag has 3 red, 5 blue, 2 green marbles. Probability of picking red?
- Favourable: 3 red
- Total: 3 + 5 + 2 = 10
- P(red) = 3/10

**Common misconception**: Students write probability as a ratio like "3 out of 10" instead of a fraction 3/10. Both communicate the same thing, but the fraction form is standard and needed for calculations.

## Complementary Events
The complement of an event A is "everything that is NOT A", written as A' or P(not A).

Rule: P(A) + P(not A) = 1, therefore P(not A) = 1 − P(A)

Example: If P(rain tomorrow) = 0.3, then P(no rain) = 1 − 0.3 = 0.7

Example: If P(drawing an ace) = 4/52 = 1/13, then P(not drawing an ace) = 1 − 1/13 = 12/13

**Common misconception**: Students add the complement and the event and get a number other than 1. If this happens, the sample space is incomplete or miscounted.

## Mutually Exclusive Events
Two events are mutually exclusive if they CANNOT both happen at the same time.

Example: Rolling a 2 and rolling a 5 on a single die — mutually exclusive.
Example: Drawing a King and drawing a Queen from one draw — mutually exclusive.
Counterexample: Drawing a King and drawing a Heart — NOT mutually exclusive (King of Hearts exists).

Addition Rule for mutually exclusive events:
P(A or B) = P(A) + P(B)

Example: P(rolling 2 or 5) = P(2) + P(5) = 1/6 + 1/6 = 2/6 = 1/3

## Non-Mutually Exclusive Events (OR Rule)
When events CAN overlap, you must subtract the overlap to avoid double-counting.

P(A or B) = P(A) + P(B) − P(A and B)

Example: P(drawing a King or a Heart)?
- P(King) = 4/52
- P(Heart) = 13/52
- P(King AND Heart) = 1/52 (King of Hearts counted in both)
- P(King or Heart) = 4/52 + 13/52 − 1/52 = 16/52 = 4/13

**Common misconception**: Simply adding P(A) + P(B) without subtracting the overlap when events are not mutually exclusive.

## Independent Events (AND Rule)
Two events are independent if the outcome of one does NOT affect the other.

Rule: P(A and B) = P(A) × P(B)

Example: Flipping a coin twice — each flip is independent.
P(Heads on flip 1 AND Heads on flip 2) = 1/2 × 1/2 = 1/4

Example: Rolling a die twice.
P(3 on roll 1 AND 5 on roll 2) = 1/6 × 1/6 = 1/36

**Common misconception**: The Gambler's Fallacy — believing that because heads came up 5 times in a row, tails is "due". Each coin flip is independent. Previous results do NOT affect the next flip. P(Heads) is always 1/2, no matter what happened before.

## Dependent Events (Conditional Probability)
When events are dependent, the outcome of one changes the probability of the next.

Example: Drawing two cards without replacement.
P(Ace on first draw) = 4/52
P(Ace on second draw | Ace on first) = 3/51 (one ace already removed)
P(both Aces) = 4/52 × 3/51 = 12/2652 = 1/221

**Common misconception**: Using the original probabilities for both draws when sampling without replacement. After the first draw, the total pool shrinks and so does the count of favourable outcomes.

## Experimental vs Theoretical Probability
- **Theoretical probability**: calculated using equally likely outcomes (what SHOULD happen)
- **Experimental probability**: calculated from actual trials (what DID happen)

Theoretical P(Heads) = 1/2
Experimental: if you flipped 100 times and got 47 heads, experimental P(Heads) = 47/100

As the number of trials increases, experimental probability gets closer to theoretical probability (Law of Large Numbers).

## Common Misconceptions Summary
1. **Adding when should multiply (AND events)**: P(A and B) = P(A) × P(B) for independent events.
2. **Forgetting to subtract overlap (OR non-mutually exclusive)**: P(A or B) = P(A) + P(B) − P(A and B).
3. **Gambler's Fallacy**: Past independent events do not affect future probabilities.
4. **Using original total after sampling without replacement**: Denominator must decrease.
5. **Probability greater than 1**: If you calculate P > 1, you made an error — recount outcomes.

## Grading Rubric — Simple Probability
CORRECT: P(event) = favourable / total; answer between 0 and 1.
PARTIAL: Correct formula but arithmetic error; correct fraction not simplified.
WRONG: Flips numerator and denominator; answer greater than 1; counts outcomes incorrectly.

## Grading Rubric — Independent Events AND
CORRECT: P(A and B) = P(A) × P(B); correctly identifies independence.
PARTIAL: Identified independence but arithmetic error.
WRONG: Adds probabilities instead of multiplying; Gambler's Fallacy reasoning.

## Grading Rubric — Mutually Exclusive OR
CORRECT: P(A or B) = P(A) + P(B) when mutually exclusive.
WRONG: Multiplies instead of adds; applies addition rule to overlapping events without subtracting.

## Grading Rubric — Non-Mutually Exclusive OR
CORRECT: P(A or B) = P(A) + P(B) − P(A and B); correctly subtracts intersection.
WRONG: Forgets to subtract the overlap; double counts.

## Grading Rubric — Dependent Events Without Replacement
CORRECT: Updates denominator after each draw; P(2nd | 1st) uses reduced pool.
WRONG: Keeps same denominator for second draw (treats as with-replacement).

## Worked Example — Simple Probability
**Problem**: A bag has 3 red, 5 blue, 2 green marbles. What is P(red)?
Step 1: Favourable = 3 red. Total = 3+5+2 = 10.
Step 2: P(red) = 3/10
**Answer: 3/10 = 0.3 = 30%**

## Worked Example — Complementary Events
**Problem**: P(rain) = 0.3. What is P(no rain)?
P(no rain) = 1 − 0.3 = **0.7**

## Worked Example — Independent Events
**Problem**: Flip two fair coins. P(both heads)?
P(H on coin 1) = 1/2; P(H on coin 2) = 1/2; coins are independent.
P(both H) = 1/2 × 1/2 = **1/4**
Do NOT add: 1/2 + 1/2 = 1 would mean certain, which is wrong.

## Worked Example — Non-Mutually Exclusive OR
**Problem**: From 52 cards, P(Heart OR King)?
P(Heart) = 13/52; P(King) = 4/52; P(Heart AND King) = 1/52 (King of Hearts).
P(Heart OR King) = 13/52 + 4/52 − 1/52 = **16/52 = 4/13**

## Worked Example — Dependent Events Without Replacement
**Problem**: Bag has 5 red, 3 blue. Pick two without replacing. P(both red)?
P(1st red) = 5/8. After removing one red: 4 red left, 7 total.
P(2nd red | 1st red) = 4/7.
P(both red) = 5/8 × 4/7 = 20/56 = **5/14**
