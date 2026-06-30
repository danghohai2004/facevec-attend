export type ShiftSettings = {
  checkInStart: string;
  checkInEnd: string;
  checkOutStart: string;
  checkOutEnd: string;
};

export type Employee = {
  id: string;
  name: string;
  emp_code: string;
  department?: string;
  created_at?: string;
  status?: "active" | "inactive";
};

export type EmployeeList = {
  items: Employee[];
  total: number;
  page: number;
  page_size: number;
};

export type AttendanceRecord = {
  id: string;
  emp_id: string;
  name?: string;
  date: string;
  check_in?: string;
  check_out?: string;
  working_hours?: number;
  status?: "checkin" | "checkout";
};
