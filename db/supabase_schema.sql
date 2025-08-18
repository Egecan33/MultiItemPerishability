-- db/supabase_schema.sql
-- Minimal schema to control runs and store results

-- Instances (store raw json + denormalized fields for quick filter)
create table
if
  not exists instances (
    id uuid primary key default gen_random_uuid()
    , created_at timestamptz not null default now()
    , name text
    , period int not null
    , manual_capacity jsonb
    , warehouse_capacity numeric
    , raw jsonb not null
  );

  -- Runs (control + status)
  create table
  if
    not exists runs (
      id uuid primary key default gen_random_uuid()
      , created_at timestamptz not null default now()
      , instance_id uuid not null references instances(id)
      on delete cascade
      , status text not null default 'queued'
      , -- queued | running | done | error
        time_limit_sec int default 3600
      , mip_gap_target numeric default 0.01
      , x_as_integer boolean default false
      , solver_version text default 'mip_lefo_v1'
      , started_at timestamptz
      , ended_at timestamptz
      , status_name text
      , objective numeric
      , best_bound numeric
      , mip_gap numeric
      , runtime_sec numeric
      , notes text
    );

    create index
    if
      not exists runs_status_idx
      on runs(status);

      -- Decisions (X arcs) – sparse rows only
      create table
      if
        not exists run_x (
          run_id uuid references runs(id)
          on delete cascade
          , item_id int not null
          , t int not null
          , u int not null
          , x numeric not null
          , primary key (run_id, item_id, t, u)
        );

        -- Setups (Y)
        create table
        if
          not exists run_y (
            run_id uuid references runs(id)
            on delete cascade
            , item_id int not null
            , t int not null
            , y int not null
            , primary key (run_id, item_id, t)
          );

          -- Simple view to check active queue
          create or replace view v_runs_queue as
          select
            *
          from
            runs
          where
            status = 'queued'
          order by
            created_at asc;

          -- Example insert flow:

          -- 1) create an instance record
          -- insert into instances(name, period, manual_capacity, warehouse_capacity, raw)
          -- values ('sample_1', 60, '[10000, ...]'::jsonb, null, :json_payload)
          -- returning id;

          -- 2) enqueue a run
          -- insert into runs(instance_id, time_limit_sec, mip_gap_target, x_as_integer)
          -- values (:instance_id, 3600, 0.01, false);

          -- 3) poll queued runs from your worker:
          -- select * from v_runs_queue limit 1;