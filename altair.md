# Import Altair

```python
import altair as alt
# if in notebook
alt.renderers.enable("notebook")
```

# Bar chart

```python
alt.Chart(df).mark_bar().encode(
    x = alt.X('column_name_x', bin = alt.Bin(maxbins=20), title="name_axis_x"),
    y="count()"
).properties(
    width = 200, 
    height= 200,
    title="name_grafic"
)
```

# scatter Plot

```python
alt.Chart(df).mark_circle().encode(
    x="column_name_x",
    y="column_name_y"
)
```

# Line Chart

```python
alt.Chart(df).mark_line().encode(
    x="date:T", # date tyme, not category
    y="sum(column_to_sum)",
    color="colum_gives_colors"
)
```

# Map

## Horizonta concatenation

```python
char_1|char_2
```

## Vertical concatenation

```python
char_1&char_2
```

# Single selection

```python
select_search_term = alt.selection_single(encodings=["x"])

trends_line = alt.Chart(df).mark_line().encode(
    x=alt.X("date:T", timeUnit="yearmonth"), 
    y="sum(column_to_sum)",
    color="colum_gives_colors"
).transform_filter(
    select_search_term
)

trends_bar = alt.Chart(df).mark_bar().encode(
    x="search_term",
    y="sum(column_to_sum)",
    color="colum_gives_colors"
).properties(
    selection=select_search_term
)

trends_bar|trends_line

```

# Interval Selection

```python
select_interval = alt.selection_interval(encodings=["x"])

trends_line = alt.Chart(df).mark_line().encode(
    x=alt.X("date:T", timeUnit="yearmonth"), 
    y="sum(column_to_sum)",
    color="column_name"
).properties(
    selection=select_interval
)

trends_bar = alt.Chart(df).mark_bar().encode(
    x="search_term",
    y="sum(column_to_sum)",
    color="column_name"
).transform_filter(
    select_interval
)

trends_bar|trends_line

```
