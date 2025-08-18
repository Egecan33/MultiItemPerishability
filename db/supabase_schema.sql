-- Enable pgcrypto for UUIDs if not enabled
create extension
if
  not exists "pgcrypto";

  -- 1) Instances (entire generator output JSON)
  create table
  if
    not exists instances (
      id uuid primary key default gen_random_uuid()
      , created_at timestamptz not null default now()
      , period int not null
      , manual_capacity jsonb
      , warehouse_capacity numeric
      , data jsonb not null
    );

    -- 2) Runs (one solve per instance)
    create table
    if
      not exists runs (
        id uuid primary key default gen_random_uuid()
        , created_at timestamptz not null default now()
        , instance_id uuid not null references instances(id)
        on delete cascade
        , time_limit_sec int
        , mip_gap numeric
        , status int
        , objective numeric
        , best_bound numeric
        , gap numeric
        , runtime_sec numeric
        , solver_version text default 'lefo_mip_v1'
      );

      -- 3) Orders (flat long table for easy charts)
      create table
      if
        not exists orders (
          run_id uuid not null references runs(id)
          on delete cascade
          , item_id int not null
          , t int not null
          , qty numeric not null
          , primary key (run_id, item_id, t)
        );

        -- Simple RLS setup (Anon can insert/select their own stuff).
        -- If you don't use auth (Anon only), you can loosen these or disable RLS.
        alter table instances
        enable row level
        security;
        alter table runs
        enable row level
        security;
        alter table orders
        enable row level
        security;

        create policy "allow anon read"
        on instances
        for
        select
        using (true);
        create policy "allow anon insert"
        on instances
        for insert
        with check (true);

        create policy "allow anon read"
        on runs
        for
        select
        using (true);
        create policy "allow anon insert"
        on runs
        for insert
        with check (true);

        create policy "allow anon read"
        on orders
        for
        select
        using (true);
        create policy "allow anon insert"
        on orders
        for insert
        with check (true);